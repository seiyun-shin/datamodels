import time
import numpy as np
from typing import List
from tqdm import tqdm

import torch as ch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torchvision
import argparse
from argparse import ArgumentParser

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (RandomHorizontalFlip, Cutout, RandomTranslate,
                             Convert, ToDevice, ToTensor, ToTorchImage)
from ffcv.transforms.common import Squeeze

################################################################################
# 1) FASTARGS SECTIONS
################################################################################
Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=0.5),
    epochs=Param(int, 'Number of epochs to run for', default=24),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', default=5),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=4),
    lr_tta=Param(bool, 'Test-time augmentation by averaging with horizontally flipped version', default=True)
)

Section('data', 'Data-related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', 
        default='/tmp/datasets/cifar_ffcv/cifar_train.beton'),
    val_dataset=Param(str, '.dat file to use for validation', 
        default='/tmp/datasets/cifar_ffcv/cifar_val.beton'),
)

Section('model', 'Model choice').params(
    arch=Param(str, 'Which model(s) to train/evaluate', default='resnet')
)

################################################################################
# 2) UTILITY FUNCTIONS
################################################################################
def sample_sparse_sign_matrix(m, n, sparsity):
    """
    Sample an m x n sparse sign matrix S where each entry is in {+1, 0, -1}.
    Probabilities:
        +1 with prob 1/(2*sparsity)
         0 with prob 1 - 1/sparsity
        -1 with prob 1/(2*sparsity)
    """
    S = np.random.choice([1, 0, -1],
                         size=(m, n),
                         p=[1/(2*sparsity), 1 - 1/sparsity, 1/(2*sparsity)])
    return S

def get_indices_from_row(row):
    """
    Given a 1D numpy array 'row' in {+1, 0, -1},
    return arrays of indices for P (row=+1),
                          N (row=-1),
                          Z (row= 0).
    """
    P = np.where(row == 1)[0]
    N = np.where(row == -1)[0]
    Z = np.where(row == 0)[0]
    return P, N, Z

################################################################################
# 3) MAKE A SINGLE DATA LOADER THAT RETURNS SAMPLE INDICES
################################################################################
@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_dataset=None, val_dataset=None,
                     batch_size=None, num_workers=None):
    """
    We'll build exactly one train loader (over the full dataset)
    and one test loader. Both return their sample indices so
    we can do subset filtering on the fly.
    """
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD  = [51.5865, 50.847,  51.255]

    # -- Pipeline for "image" field
    #    These are standard FFCV transforms for CIFAR-10
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
    image_pipeline.extend([
        ToTensor(),
        ToDevice(ch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        Convert(ch.float16),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    # For training, we'll add random augmentations inside the loop below

    # -- Pipeline for "label" field
    label_pipeline: List[Operation] = [
        IntDecoder(), ToTensor(), ToDevice(ch.device("cuda:0")), Squeeze()
    ]

    # -- We also define a pipeline for an "index" field
    #    Typically you need your .beton to have a custom field that stores the sample index.
    #    If not, you can use "SampleIndicesDecoder()" in the newest versions of FFCV.
    #    We'll assume we have an IntDecoder for "indices".
    #    Adjust as needed for your FFCV setup.
    index_pipeline: List[Operation] = [
        IntDecoder(),  # or SampleIndicesDecoder() in newer FFCV
        ToTensor(),
        ToDevice(ch.device("cuda:0"), non_blocking=True)
    ]

    # 3a) Build the train loader
    train_image_pipeline = [
        SimpleRGBImageDecoder(),
        RandomHorizontalFlip(),
        RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
        Cutout(4, tuple(map(int, CIFAR_MEAN))),
        ToTensor(),
        ToDevice(ch.device("cuda:0"), non_blocking=True),
        ToTorchImage(),
        Convert(ch.float16),
        torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]

    train_loader = Loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.RANDOM,
        drop_last=True,
        custom_fields={
            'index': IntDecoder()  # or SampleIndicesDecoder()
        },
        pipelines={
            'image': train_image_pipeline,
            'label': label_pipeline,
            'index': index_pipeline
        }
    )

    # 3b) Build the test loader
    test_loader = Loader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        drop_last=False,
        custom_fields={
            'index': IntDecoder()  # or SampleIndicesDecoder()
        },
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline,
            'index': index_pipeline
        }
    )

    loaders = {
        'train': train_loader,
        'test':  test_loader
    }
    return loaders

################################################################################
# 4) MODEL DEFINITION (same as your existing code)
################################################################################
class Mul(ch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x):
        return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x):
        return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
        ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                     stride=stride, padding=padding, groups=groups, bias=False),
        ch.nn.BatchNorm2d(channels_out),
        ch.nn.ReLU(inplace=True)
    )

def construct_model(model_type="resnet"):
    num_class = 10
    model = ch.nn.Sequential(
        conv_bn(3,   64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.cuda().to(memory_format=ch.channels_last)
    return model

################################################################################
# 5) TRAINING / EVALUATION WITH ON-THE-FLY MASKING
################################################################################
@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
def train_masked(model, train_loader, mask_set,
                 lr=None, epochs=None, label_smoothing=None,
                 momentum=None, weight_decay=None, lr_peak_epoch=None):
    """
    Trains 'model' for 'epochs' on exactly the samples whose index is in 'mask_set'.
    We do NOT recreate the loader; we just skip anything not in the mask.
    """
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    iters_per_epoch = len(train_loader)
    # Cyclic LR schedule (single triangle)
    lr_schedule = np.interp(
        np.arange((epochs+1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0]
    )
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)

    scaler = GradScaler("cuda")
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    model.train()
    step_count = 0

    for _ in range(epochs):
        # "train_loader" yields (images, labels, idxs)
        for ims, labs, idxs in tqdm(train_loader, leave=False):
            # 1) Subset the current batch to items in mask_set
            #    idxs is a Tensor on GPU. Convert to CPU numpy or check membership in a GPU set.
            idxs_cpu = idxs.cpu().numpy()
            keep = [i for i, sample_idx in enumerate(idxs_cpu) if sample_idx in mask_set]

            if len(keep) == 0:
                # No samples from this batch belong to mask_set; skip
                continue

            # 2) Gather the relevant samples
            ims_sub  = ims[keep]
            labs_sub = labs[keep]

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                out = model(ims_sub)
                loss = loss_fn(out, labs_sub)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            # Because we do one LR step per iteration
            scheduler.step()
            step_count += 1

@param('training.lr_tta')
def evaluate(model, test_loader, lr_tta=False):
    """
    Evaluate on the entire test set (no masking).
    """
    model.eval()
    all_margins = []
    with ch.no_grad():
        for ims, labs, idxs in tqdm(test_loader, leave=False):
            with autocast(device_type="cuda"):
                out = model(ims)
                if lr_tta:
                    out += model(ch.fliplr(ims))
                    out /= 2

                # margin calculation
                class_logits = out[ch.arange(out.shape[0]), labs].clone()
                out[ch.arange(out.shape[0]), labs] = -1000
                next_classes = out.argmax(1)
                class_logits -= out[ch.arange(out.shape[0]), next_classes]

            all_margins.append(class_logits.cpu())

    all_margins = ch.cat(all_margins)
    print('Average margin:', all_margins.mean())
    return all_margins.numpy()

################################################################################
# 6) TRAIN TWO MODELS FOR A ROW, BUT DO *NOT* BUILD NEW LOADERS
################################################################################
def train_two_models_for_row(row_idx,
                             row_vals,
                             train_loader,
                             test_loader,
                             model_constructor,
                             device='cuda',
                             s=100,
                             n=50000):
    """
    For row i, define P_i, N_i, Z_i. Train:
      - model_pz on P_i ∪ Z_i
      - model_nz on N_i ∪ Z_i
    Then measure difference of their margins on the test set.
    """
    # 1) figure out P_i, N_i, Z_i
    P, N, Z = get_indices_from_row(row_vals)

    pz_set = set(np.union1d(P, Z))  # Python set for O(1) membership checks
    nz_set = set(np.union1d(N, Z))

    # 2) Construct 2 fresh models
    model_pz = model_constructor().to(device)
    train_masked(model_pz, train_loader, pz_set)

    model_nz = model_constructor().to(device)
    train_masked(model_nz, train_loader, nz_set)

    # 3) Evaluate both on test set
    out_pz = evaluate(model_pz, test_loader)
    out_nz = evaluate(model_nz, test_loader)

    # 4) difference
    diff_val = (out_pz.mean() - out_nz.mean())

    # 5) scale by sqrt(s/n)
    yi = np.sqrt(s / n) * diff_val
    return yi

################################################################################
# 7) MAIN ENTRY POINT
################################################################################
def main(index, logdir):
    """
    This "main" creates one train loader and one test loader (both full),
    then for each row i in S, trains two specialized models by filtering
    the train loader on the fly (no repeated loader creation).
    """
    config = get_current_config()
    parser = argparse.ArgumentParser(description='Fast CIFAR-10 + Compressed Sensing')

    parser.add_argument("--model.arch", type=str,
        choices=['resnet', 'vit', 'sparse_vit', 'all'],
        default='resnet',
        help="Which model(s) to train/evaluate."
    )
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    device = 'cuda'
    # Example usage of config:
    lr     = config['training.lr']
    epochs = config['training.epochs']

    # We'll sample a matrix S ∈ {+1, 0, -1}^{m×n}
    n = 50000   # e.g. total CIFAR-10 train size
    s = 100     
    m = 100     


    # (1) Sample S
    S = sample_sparse_sign_matrix(m=m, n=n, sparsity=sparsity)
    # (2) A = sqrt(s/n)*S
    A = np.sqrt(float(s)/n) * S

    # (3) Make ONE train loader & ONE test loader
    loaders = make_dataloaders()
    train_loader = loaders['train']
    test_loader  = loaders['test']

    # (4) Decide how to build new models
    def model_constructor():
        arch = config['model.arch']
        if arch == 'resnet':
            return construct_model("resnet")
        elif arch == 'vit':
            return construct_model("vit")
        elif arch == 'sparse_vit':
            return construct_model("sparse_vit")
        else:
            # default
            return construct_model("resnet")

    # (5) For each row i, train two masked models
    y = np.zeros(m, dtype=np.float32)
    for i in range(m):
        row_vals = S[i]
        y[i] = train_two_models_for_row(
            row_idx=i,
            row_vals=row_vals,
            train_loader=train_loader,
            test_loader=test_loader,
            model_constructor=model_constructor,
            device=device,
            s=s,
            n=n
        )
        print(f"[Row {i}/{m}]  y[i] = {y[i]}")

    print("Measurement vector y shape:", y.shape)
    print("Done with compressed-sensing procedure!")
    return {'masks': A, 'margins': y}
