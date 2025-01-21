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

# ---------------------------
# Fastargs sections & params
# ---------------------------
Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=0.5),
    epochs=Param(int, 'Number of epochs to run for', default=24),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', default=5),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=4),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True)
)

Section('data', 'Data-related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', 
        default='/tmp/datasets/cifar_ffcv/cifar_train.beton'),
    val_dataset=Param(str, '.dat file to use for validation', 
        default='/tmp/datasets/cifar_ffcv/cifar_val.beton'),
)

# NEW: add a section for the model choice
Section('model', 'Model choice').params(
    arch=Param(str, 'Which model(s) to train/evaluate', default='resnet')
)

# ---------------------------
# Utility functions
# ---------------------------
def sample_sparse_sign_matrix(m, n, sparsity):
    """
    Sample an m x n sparse sign matrix S where each entry is in {+1, 0, -1},
    using probabilities:
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
    Given a 1D numpy array 'row' in {+1,0,-1},
    return arrays of indices for P (where row=+1),
    N (where row=-1), and Z (where row=0).
    """
    P = np.where(row == 1)[0]
    N = np.where(row == -1)[0]
    Z = np.where(row == 0)[0]
    return P, N, Z

@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_dataset=None, val_dataset=None,
                     batch_size=None, num_workers=None, mask=None):
    paths = {
        'train': train_dataset,
        'test':  val_dataset
    }

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD  = [51.5865, 50.847,  51.255]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [
            IntDecoder(), ToTensor(), ToDevice(ch.device("cuda:0")), Squeeze()
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device("cuda:0"), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(
            paths[name],
            indices=(mask if name == 'train' else None),
            batch_size=batch_size,
            num_workers=num_workers,
            order=ordering,
            drop_last=(name == 'train'),
            pipelines={'image': image_pipeline, 'label': label_pipeline}
        )

    return loaders

# Simple model components
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
    """
    Extend this if you really need different architectures (resnet/vit/etc.).
    For now, all calls just build the same example CNN.
    """
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

@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
def train(model, loaders, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(
        np.arange((epochs+1) * iters_per_epoch),
        [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
        [0, 1, 0]
    )
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler("cuda")
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for _ in range(epochs):
        for ims, labs in tqdm(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

@param('training.lr_tta')
def evaluate(model, loaders, lr_tta=False):
    model.eval()
    with ch.no_grad():
        all_margins = []
        for ims, labs in tqdm(loaders['test']):
            with autocast(device_type="cuda"):
                out = model(ims)
                if lr_tta:
                    out += model(ch.fliplr(ims))
                    out /= 2
                class_logits = out[ch.arange(out.shape[0]), labs].clone()
                out[ch.arange(out.shape[0]), labs] = -1000
                next_classes = out.argmax(1)
                class_logits -= out[ch.arange(out.shape[0]), next_classes]
                all_margins.append(class_logits.cpu())
        all_margins = ch.cat(all_margins)
        print('Average margin:', all_margins.mean())
        return all_margins.numpy()

def train_two_models_for_row(row_idx,
                             row_vals,
                             loaders,
                             model_constructor,
                             device='cuda',
                             s=50000/3,
                             n=50000):
    """
    row_idx: which row we are dealing with (for logging/debug)
    row_vals: 1D numpy array in {+1,0,-1}, length n
    loaders: dictionary of {'train':some_loader,'test':some_loader}
    model_constructor: function that constructs a fresh model
    s, n: used in the sqrt(s/n) factor
    """
    # 1) figure out P_i, N_i, Z_i
    P, N, Z = get_indices_from_row(row_vals)

    # combine arrays using union1d or concatenate
    pz_mask = np.union1d(P, Z)  
    nz_mask = np.union1d(N, Z)

    loaders_pz = make_dataloaders(
        train_dataset=get_current_config()['data.train_dataset'],
        val_dataset=get_current_config()['data.val_dataset'],
        batch_size=get_current_config()['training.batch_size'],
        num_workers=get_current_config()['training.num_workers'],
        mask=pz_mask
    )

    loaders_nz = make_dataloaders(
        train_dataset=get_current_config()['data.train_dataset'],
        val_dataset=get_current_config()['data.val_dataset'],
        batch_size=get_current_config()['training.batch_size'],
        num_workers=get_current_config()['training.num_workers'],
        mask=nz_mask
    )

    # 3) Train two separate models
    model_pz = model_constructor().to(device)
    train(model_pz, loaders_pz)

    model_nz = model_constructor().to(device)
    train(model_nz, loaders_nz)

    # 4) Evaluate on test set
    out_pz = evaluate(model_pz, loaders)  # shape (#test_examples,)
    out_nz = evaluate(model_nz, loaders)  # shape (#test_examples,)

    diff_val = (out_pz.mean() - out_nz.mean())

    # 5) scale by sqrt(s/n)
    yi = np.sqrt(s / n) * diff_val
    return yi

def main(index, logdir):
    """
    Example main() in train_cifar_compressed_sensing.py.
    Accepts 'index' and 'logdir' so it matches your worker's call signature.
    """
    config = get_current_config()
    parser = argparse.ArgumentParser(description='Fast CIFAR-10 training + Compressed Sensing')
    
    # We'll still accept --model on the CLI, but it now maps to the fastargs param "model.arch"
    parser.add_argument("--model.arch", type=str,
        choices=['resnet', 'vit', 'sparse_vit', 'all'],
        default='resnet',
        help="Which model(s) to train and evaluate.")

    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    device = 'cuda'
    lr     = config['training.lr']
    epochs = config['training.epochs']

    # Number of training examples
    n = 50000   # e.g. total CIFAR-10 train size
    s = 50000/3     
    m = 100     

    # If we want ~100 nonzeros on average, then sparsity = n/s
    # sparsity = n / s
    sparsity = 3

    # 1) Sample S in {+1,0,-1}, shape (m,n)
    S = sample_sparse_sign_matrix(m=m, n=n, sparsity=sparsity)
    # 2) A = sqrt(s/n)*S
    A = np.sqrt(float(s)/n) * S

    # 3) Make data loaders for full dataset
    loaders = make_dataloaders()

    # NEW: read the 'model.arch' param from config
    def model_constructor():
        which = config['model.arch']
        if which == 'resnet':
            return construct_model("resnet")
        elif which == 'vit':
            return construct_model("vit")
        elif which == 'sparse_vit':
            return construct_model("sparse_vit")
        elif which == 'all':
            # default to resnet, or any custom logic
            return construct_model("resnet")
        else:
            return construct_model("resnet")

    # 4) For each row i, train two models & measure y_i
    y = np.zeros(m, dtype=np.float32)
    for i in range(m):
        row_vals = S[i]  # shape (n,)
        y_i = train_two_models_for_row(
            row_idx=i,
            row_vals=row_vals,
            loaders=loaders,
            model_constructor=model_constructor,
            device=device,
            s=s, 
            n=n
        )
        y[i] = y_i
        print(f"Row {i}/{m}: measurement y_i = {y_i}")

    print("Collected measurement vector y of shape:", y.shape)
    print("Compressed-sensing approach run complete!")

    return {
        'masks': A,
        'margins': y
    }
