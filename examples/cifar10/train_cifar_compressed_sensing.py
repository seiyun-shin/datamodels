from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torchvision
import argparse
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

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

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', 
        default='/tmp/datasets/cifar_ffcv/cifar_train.beton'),
    val_dataset=Param(str, '.dat file to use for validation', 
        default='/tmp/datasets/cifar_ffcv/cifar_val.beton'),
)
def sample_sparse_sign_matrix(m, n, sparsity):
    """
    Sample an m x n sparse sign matrix S where each entry is in {+1, 0, -1},
    using probabilities:
        +1 with prob 1/(2*sparsity)
         0 with prob 1 - 1/sparsity
        -1 with prob 1/(2*sparsity)
    """
    # Example: p=[1/(2*sparsity), 1 - 1/sparsity, 1/(2*sparsity)]
    S = np.random.choice([1, 0, -1],
                         size=(m, n),
                         p=[1/(2*sparsity), 1 - 1/sparsity, 1/(2*sparsity)])
    return S

def get_indices_from_row(row):
    """
    Given a 1D numpy array 'row' in {+1,0,-1},
    return sets (or lists) of indices for P (where row=+1),
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
def make_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=None, mask=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device("cuda:0")), Squeeze()]
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

        loaders[name] = Loader(paths[name], indices=(mask if name == 'train' else None),
                               batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders

# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
class Mul(ch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )

def construct_model():
    num_class = 10
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
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
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
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
                             s=100,
                             n=50000):
    """
    row_idx: which row we are dealing with (for logging/debug)
    row_vals: 1D numpy array in {+1,0,-1}, length n
    loaders: dictionary of {'train': some_loader, 'test': some_loader}
    model_constructor: function that constructs a fresh model (e.g. ResNet or ViT)
    s, n: used in the sqrt(s/n) factor
    """

    # 1) figure out P_i, N_i, Z_i
    P, N, Z = get_indices_from_row(row_vals)
    
    # 2) Build subsets for training:
    #    We might create specialized ffcv/Loaders, or just filter on-the-fly.
    #    For simplicity, let's define quick "subloaders" that only yield
    #    data from the desired subset of indices.
    #    If your code already uses mask=..., that’s one approach. Or you can
    #    adapt your existing make_dataloaders to accept custom subsets.
    #    Pseudocode:

    loaders_pz = make_dataloaders(
        train_dataset=get_current_config()['data.train_dataset'],
        val_dataset=get_current_config()['data.val_dataset'],
        batch_size=get_current_config()['training.batch_size'],
        num_workers=get_current_config()['training.num_workers'],
        mask=P.union(Z)  # or np.concatenate([P, Z])
    )

    loaders_nz = make_dataloaders(
        train_dataset=get_current_config()['data.train_dataset'],
        val_dataset=get_current_config()['data.val_dataset'],
        batch_size=get_current_config()['training.batch_size'],
        num_workers=get_current_config()['training.num_workers'],
        mask=N.union(Z)  # or np.concatenate([N, Z])
    )

    # 3) Train two separate models
    model_pz = model_constructor().to(device)
    train(model_pz, loaders_pz)

    model_nz = model_constructor().to(device)
    train(model_nz, loaders_nz)

    # 4) Evaluate on test set (or x_test),
    #    get difference in predictions for y_i
    out_pz = evaluate(model_pz, loaders)   # shape (#test_examples,)
    out_nz = evaluate(model_nz, loaders)   # shape (#test_examples,)

    # for simplicity, let's take the difference in average margins
    # or you might store the entire vector difference if you prefer
    diff_val = (out_pz.mean() - out_nz.mean())

    # 5) scale by sqrt(s/n)
    yi = np.sqrt(s / n) * diff_val
    return yi

def main(index, logdir):
    """
    Example main() in train_cifar.py that integrates the
    "tentative procedure" from your attached docs.
    """
    config = get_current_config()
    parser = argparse.ArgumentParser(description='Fast CIFAR-10 training + Compressed Sensing')
    parser.add_argument("--model", type=str,
                        choices=['resnet', 'vit', 'sparse_vit', 'all'],
                        default='all',
                        help="Which model(s) to train and evaluate.")
    # Possibly add more args for 'm', 'sparsity', etc.
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    # Basic hyperparams
    device = 'cuda'
    lr = config['training.lr']
    epochs = config['training.epochs']
    # etc.

    # Number of training examples
    n = 50000   # e.g. total CIFAR-10 train size
    # Suppose we fix s=100
    s = 100
    # Suppose we set m=200 as #rows in S (just an example)
    m = 200
    # "sparsity" param from your formula ~ 1/(2*sparsity) chance for +1, etc.
    #  or set it so that expected #nonzeros in row is s...
    #  e.g. expected #nonzeros = n * 1/sparsity
    #  so if we want ~ s=100 nonzeros, we can do 'sparsity = n/s'
    sparsity = n / s

    # 1) Sample S in {+1,0,-1}, shape (m,n)
    S = sample_sparse_sign_matrix(m=m, n=n, sparsity=sparsity)

    # 2) We'll define A = sqrt(s/n)*S
    A = np.sqrt(float(s)/n) * S

    # 3) Make your usual data loaders for the full dataset:
    loaders = make_dataloaders()

    # 4) For each row i in [0..m-1], define Pi, Ni, Zi, train two models, measure y_i
    #    We'll store them in a measurement vector y of length m
    y = np.zeros(m, dtype=np.float32)

    # Decide a model constructor function based on config
    def model_constructor():
        if config['model'] == 'resnet':
            return construct_model("resnet")
        elif config['model'] == 'vit':
            return construct_model("vit")
        elif config['model'] == 'sparse_vit':
            return construct_model("sparse_vit")
        else:
            # default or your choice
            return construct_model("resnet")

    for i in range(m):
        row_vals = S[i]     # shape (n,)
        # Train two models, get y_i
        y_i = train_two_models_for_row(
            row_idx=i,
            row_vals=row_vals,
            loaders=loaders,
            model_constructor=model_constructor,
            device=device,
            s=s, n=n
        )
        y[i] = y_i
        print(f"Row {i}/{m}: measurement y_i = {y_i}")

    print("Collected measurement vector y of shape:", y.shape)

    # 5) (Optional) Reconstruct x from y = A x:
    #    e.g. x_hat = l1_minimize(A, y). Then x_hat is your approximate solution
    #    This is exactly the typical "compressed sensing" step:
    #       min_x || x ||_1  s.t.  || A x - y || <= eta
    #    For a quick test:
    # from fast_l1.solver import l1_minimize
    # x_hat = l1_minimize(A, y, lam=0.1, max_iter=1000)
    # print("Recovered x with shape:", x_hat.shape)

    # 6) Save results
    # np.savez(f"{config.logdir}/cs_results.npz",
    #          A=A, S=S, y=y
    #          # x_hat=x_hat if you reconstruct
    #          )
    print("Compressed-sensing approach run complete!")
    return {
        'masks': A,
        'margins': y
    }
