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

########################################
# 1) FASTARGS SECTIONS & PARAMS
########################################
Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=0.5),
    epochs=Param(int, 'Number of epochs to run for', default=24),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', default=5),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'Number of workers', default=4),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True)
)

Section('data', 'Data-related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', 
        default='/tmp/datasets/cifar_ffcv/cifar_train.beton'),
    val_dataset=Param(str, '.dat file to use for validation', 
        default='/tmp/datasets/cifar_ffcv/cifar_val.beton'),
)

########################################
# 2) UTILITY FUNCTIONS
########################################
def get_indices_from_row(row):
    """
    Given 1D array 'row' in {+1, 0, -1}, return arrays of indices for P, N, Z.
    """
    P = np.where(row == 1)[0]
    N = np.where(row == -1)[0]
    Z = np.where(row == 0)[0]
    return P, N, Z

########################################
# 3) DATA LOADERS
########################################
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

########################################
# 4) MODEL DEFINITION
########################################
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

def construct_model():
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

########################################
# 5) TRAIN/EVAL
########################################
@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
def train(model, loaders, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None):
    from torch.optim import SGD
    from torch.optim import lr_scheduler
    from torch.amp import GradScaler, autocast
    from torch.nn import CrossEntropyLoss

    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train'])
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
                    out += model(ch.flip(ims, dims=[3]))  # horizontal flip
                    out /= 2
                class_logits = out[ch.arange(out.shape[0]), labs].clone()
                out[ch.arange(out.shape[0]), labs] = -1000
                next_classes = out.argmax(1)
                margin = class_logits - out[ch.arange(out.shape[0]), next_classes]
            all_margins.append(margin.cpu())
        all_margins = ch.cat(all_margins)
        print('Average margin:', all_margins.mean())
        return all_margins.numpy()

########################################
# 6) TRAIN TWO MODELS PER ROW, RETURN DIFF VECTOR
########################################
def train_two_models_for_row(row_idx,
                             row_vals,
                             loaders,
                             model_constructor,
                             device='cuda',
                             s=50000/3,
                             n=50000):
    """
    Returns a PER-TEST-SAMPLE difference vector instead of a scalar.
    """
    P, N, Z = get_indices_from_row(row_vals)
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

    model_pz = model_constructor().to(device)
    train(model_pz, loaders_pz)

    model_nz = model_constructor().to(device)
    train(model_nz, loaders_nz)

    out_pz = evaluate(model_pz, loaders)  # shape (#test_examples,)
    out_nz = evaluate(model_nz, loaders)  # shape (#test_examples,)

    diff_vec = out_pz - out_nz
    diff_vec = np.sqrt(s / n) * diff_vec
    print(f"[Row {row_idx}] diff_vec shape = {diff_vec.shape}, mean={diff_vec.mean():.4f}")
    return diff_vec

########################################
# 7) MAIN: LOOP OVER M ROWS
########################################
def main(index, logdir, store_dir=None):
    """
    Each worker i loads the same S from logdir, picks row i => S[i],
    then writes results to margins and sets completed[i] = True.
    """
    config = get_current_config()
    parser = argparse.ArgumentParser(description='Fast CIFAR-10 training + Compressed Sensing')
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
    m = 500  

    # 1) Load the globally generated arrays from log_dir
    #    E.g. S.npy, margins.npy, completed.npy
        # Use store_dir for the location of S
    if store_dir is None:
        store_dir = logdir  # fallback if not provided

    S_path = f"{store_dir}/S.npy"  # <<--- now it references the top-level
    print(f"Loading S from {S_path}")
    S_path       = f"{store_dir}/S.npy"
    margins_path = f"{store_dir}/margins.npy"
    completed_path = f"{store_dir}/completed.npy"

    print(f"[Worker {index}] Loading S from {S_path}")
    S = np.load(S_path, mmap_mode='r+')

    margins_array   = np.load(margins_path,   mmap_mode='r+')
    completed_array = np.load(completed_path, mmap_mode='r+')
    
    # 2) A = sqrt(s/n)*S  (We might not strictly need A if we store the diff vectors)
    A = np.sqrt(float(s)/n) * S

    # 3) Grab row i
    row_vals = S[index]  # shape (n,)
    print(f"[Worker {index}] loaded row {index}, row_vals shape={row_vals.shape}")

    # 4) Build full loaders
    loaders = make_dataloaders()

    # 5) Train subsets for that row
    diff_vec = train_two_models_for_row(
        row_idx=index,
        row_vals=row_vals,
        loaders=loaders,
        model_constructor=construct_model,
        device=device,
        s=s,
        n=n
    )

    # 5) Write diff_vec into margins_array[index]
    margins_array[index, :] = diff_vec

    # 6) Mark completed
    completed_array[index] = True

    # (In pure NumPy, changes to a memmapped array are auto-synced. 
    #  But you might call .flush() or re-save to persist on disk.)
    # e.g.:
    margins_array.flush()
    completed_array.flush()

    print(f"[Worker {index}] done, wrote margins row={index}, shape={diff_vec.shape}")
    return {
        'masks': A[index],
        'margins': diff_vec
    }
