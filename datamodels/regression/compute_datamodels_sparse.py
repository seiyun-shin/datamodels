import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch as ch

from fastargs import Param, Section, get_current_config
from fastargs.decorators import param, section

from ffcv.fields.decoders import IntDecoder, NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Squeeze, ToDevice, ToTensor
from ffcv.pipeline.operation import Operation

from fast_l1 import regressor
from dataclasses import replace


# Define operation for slicing tensors
class Slice(Operation):
    def __init__(self, start_ind, end_ind) -> None:
        super().__init__()
        self.start_ind = start_ind
        self.end_ind = end_ind

    def generate_code(self):
        start_ind = self.start_ind
        end_ind = self.end_ind

        def make_slice(inp, _):
            if end_ind == -1:
                return inp[:, start_ind:]
            return inp[:, start_ind:end_ind]

        return make_slice

    def declare_state_and_memory(self, previous_state):
        end_ind = previous_state.shape[0] if self.end_ind == -1 else self.end_ind
        new_shape = (int(end_ind) - self.start_ind,)
        return replace(previous_state, shape=new_shape), None


# Define sections for parameter configurations
Section('data', 'source data info').params(
    data_path=Param(str, 'Path to beton file', required=True),
    num_train=Param(int, 'Number of models for training', required=True),
    num_val=Param(int, 'Number of models for validation', required=True),
    sparsity=Param(int, 'Sparsity level for compressed sensing matrix', default=10),
    num_classes=Param(int, 'Number of classes for multi-class classification', required=True),
)

Section('cfg', 'arguments to give the writer').params(
    k=Param(int, 'Number of lambdas on the regularization path', required=True),
    lr=Param(float, 'Learning rate to use', default=0.01),
    eps=Param(float, '(min lambda) / (max lambda)', default=1e-5),
    batch_size=Param(int, 'Batch size for regression', required=True),
    out_dir=Param(str, 'Where to write', required=True),
    num_workers=Param(int, 'Num of workers to use for dataloading', default=16),
    use_bias=Param(int, 'Whether to use the bias parameter', default=1)
)


# Generate loaders for train, validation, and test datasets
@param('data.data_path')
@param('data.num_train')
@param('data.num_val')
def make_loaders(data_path=None, num_train=None, num_val=None):
    train_loader = make_loader(np.arange(num_train), data_path)
    val_loader = make_loader(np.arange(num_train, num_train + num_val), data_path)
    return train_loader, val_loader


@param('data.data_path')
@param('cfg.num_workers')
@param('cfg.batch_size')
def make_loader(indices, data_path=None, num_workers=None, batch_size=None):
    return Loader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.RANDOM,
        indices=indices,
        os_cache=True,
        pipelines={
            'mask': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'))],
            'targets': [NDArrayDecoder(), ToTensor(), ToDevice(ch.device('cuda:0'))],
            'idx': [IntDecoder(), ToTensor(), Squeeze(), ToDevice(ch.device('cuda:0'))]
        },
        recompile=False
    )

# Construct sparse sign matrix
def sample_sparse_sign_matrix(m, n, sparsity):
    S = np.random.choice([1, 0, -1], size=(m, n), p=[1 / (2 * sparsity), 1 - 1 / sparsity, 1 / (2 * sparsity)])
    return ch.tensor(S, dtype=ch.float32).cuda()


# Construct design matrix A
def construct_design_matrix(m, n, sparsity):
    S = sample_sparse_sign_matrix(m, n, sparsity)
    A = np.sqrt(sparsity / n) * S
    return A


# Main function
@param('data.data_path')
@param('data.num_train')
@param('data.num_val')
@param('data.num_classes')
@param('data.sparsity')
@param('cfg.k')
@param('cfg.lr')
@param('cfg.eps')
@param('cfg.out_dir')
def main(data_path, num_train, num_val, num_classes, sparsity, k, lr, eps, out_dir):
    train_loader, val_loader = make_loaders(data_path, num_train, num_val)

    # Initialize model parameters
    n_features = train_loader.reader.handlers['mask'].shape[0]
    weight = ch.zeros(n_features, num_classes).cuda()  # One weight vector per class
    bias = ch.zeros(num_classes).cuda()  # One bias per class

    # Calculate maximum lambda
    max_lam = regressor.calc_max_lambda(train_loader)

    # Generate compressed sensing matrix
    A = construct_design_matrix(num_train, n_features, sparsity)

    # Prepare logging directory
    logdir = Path(out_dir) / "logs/compressed_sensing/"
    os.makedirs(logdir, exist_ok=True)

    # Train model using compressed sensing
    regressor.train_saga_with_cs(
        weight,
        bias,
        train_loader,
        val_loader,
        A=A,
        lr=lr,
        start_lams=max_lam,
        num_lambdas=k,
        lam_decay=np.exp(np.log(eps) / k),
        logdir=str(logdir)
    )
    print("Training completed.")


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description="Datamodel regression with compressed sensing for multi-class classification")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
