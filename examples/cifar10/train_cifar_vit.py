from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import ViTForImageClassification
import torchvision

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze


from ffcv.pipeline.operation import Operation
import numpy as np
from PIL import Image

class ResizeWrapper(Operation):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def generate_code(self):
        def resize(images, *args):
            resized_images = np.zeros((images.shape[0], self.size[1], self.size[0], images.shape[3]), dtype=images.dtype)
            for i in range(images.shape[0]):
                img = Image.fromarray(images[i])
                img = img.resize(self.size, Image.BICUBIC)
                resized_images[i] = np.array(img)
            return resized_images
        return resize

# Construct sparse sign matrix
def sample_sparse_sign_matrix(m, n, sparsity):
    """
    Generate a sparse sign matrix S of shape (m, n) with sparsity.
    """
    S = np.random.choice([1, 0, -1], size=(m, n), p=[1 / (2 * sparsity), 1 - 1 / sparsity, 1 / (2 * sparsity)])
    return ch.tensor(S, dtype=ch.float32).cuda()


# Construct design matrix A
def construct_design_matrix(m, n, sparsity):
    """
    Construct the design matrix using the sparse sign matrix.
    """
    S = sample_sparse_sign_matrix(m, n, sparsity)
    A = np.sqrt(sparsity / n) * S
    return A


Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=5e-5),
    epochs=Param(int, 'Number of epochs to run for', default=20),
    batch_size=Param(int, 'Batch size', default=64),
    weight_decay=Param(float, 'L2 weight decay', default=0.01),
    num_workers=Param(int, 'The number of workers', default=4),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True),
    sparsity=Param(int, 'Sparsity level for compressed sensing', default=10)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', 
        default='/tmp/datasets/cifar_ffcv/cifar_train.beton'),
    val_dataset=Param(str, '.dat file to use for validation', 
        default='/tmp/datasets/cifar_ffcv/cifar_val.beton'),
)


from ffcv.transforms import SimpleRGBImageDecoder, ToDevice, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.transforms import ToTorchImage

@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    # Generate sparse sign matrix for compressed sensing
    total_samples = 50_000  # CIFAR-10 training dataset size
    sparsity = 100  # Number of "influential" samples in total
    num_rows = int(np.ceil((sparsity * np.log(2 * total_samples / sparsity) + np.log(100)) / (0.465 ** 2)))

    sparse_matrix = construct_design_matrix(num_rows, total_samples, sparsity)
    mask = (sparse_matrix.sum(axis=1) > 0).to(dtype=ch.bool)  # Convert to boolean tensor

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device("cuda:0")), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        if name == 'train':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
                ResizeWrapper((224, 224)),  # Add Resize transformation using ResizeWrapper
            ])
        else:
            image_pipeline.extend([
                ResizeWrapper((224, 224)),  # Resize for test data as well
            ])

        ordering = OrderOption.RANDOM if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name], indices=(np.nonzero(mask.cpu().numpy())[0] if name == 'train' else None),
                               batch_size=batch_size, num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'),
                               pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders

def construct_vit_model():
    # Load pretrained ViT model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=10  # CIFAR-10 has 10 classes
    ).cuda()
    
    # Reinitialize the classifier head for CIFAR-10
    model.classifier = ch.nn.Linear(
        in_features=model.config.hidden_size,  # Match hidden size from pretrained config
        out_features=10  # CIFAR-10 has 10 classes
    ).cuda()
    
    # Ensure new classifier weights are initialized
    ch.nn.init.xavier_uniform_(model.classifier.weight)
    ch.nn.init.zeros_(model.classifier.bias)
    
    return model


@param('training.lr')
@param('training.epochs')
@param('training.weight_decay')
def train(model, loaders, lr=None, epochs=None, weight_decay=None):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda")
    loss_fn = CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for ims, labs in tqdm(loaders['train']):
            opt.zero_grad()
            with autocast(device_type="cuda"):
                out = model(ims).logits  # For ViT models
                loss = loss_fn(out, labs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()


@param('training.lr_tta')
def evaluate(model, loaders, lr_tta=False):
    model.eval()
    with ch.no_grad():
        all_margins = []
        for ims, labs in tqdm(loaders['test']):
            with autocast(device_type="cuda"):
                out = model(ims).logits  # For ViT models
                if lr_tta:
                    out += model(ch.fliplr(ims)).logits
                    out /= 2
                class_logits = out[ch.arange(out.shape[0]), labs].clone()
                out[ch.arange(out.shape[0]), labs] = -1000
                next_classes = out.argmax(1)
                class_logits -= out[ch.arange(out.shape[0]), next_classes]
                all_margins.append(class_logits.cpu())
        all_margins = ch.cat(all_margins)
        print('Average margin:', all_margins.mean())
        return all_margins.numpy()


def main(index, logdir):
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 ViT training with compressed sensing')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders = make_dataloaders()
    model = construct_vit_model()
    train(model, loaders)
    res = evaluate(model, loaders)
    print(mask.shape, res.shape)
    return {
        'masks': mask,
        'margins': res
    }
