import argparse
import time
import numpy as np
from tqdm import tqdm

import torch as ch
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
import torchvision

from transformers import ViTForImageClassification

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param

from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=5e-5),
    epochs=Param(int, 'Number of epochs to run for', default=20),
    batch_size=Param(int, 'Batch size', default=64),
    weight_decay=Param(float, 'L2 weight decay', default=0.01),
    num_workers=Param(int, 'The number of workers', default=4),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True)
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.dat file to use for training', 
        default='/tmp/datasets/cifar_ffcv/cifar_train.beton'),
    val_dataset=Param(str, '.dat file to use for validation', 
        default='/tmp/datasets/cifar_ffcv/cifar_val.beton'),
)

@param('data.train_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
def make_dataloaders(train_dataset=None, val_dataset=None, batch_size=None, num_workers=None, mask=None):
    paths = {
        'train': train_dataset,
        'test': val_dataset
    }

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

def construct_model(model_type="resnet"):
    if model_type == "resnet":
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
    elif model_type == "vit":
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k", num_labels=10
        ).cuda()
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
                out = model(ims)
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

# def main(index, logdir):
#     config = get_current_config()
#     parser = argparse.ArgumentParser(description='Fast CIFAR-10 training')
#     parser.add_argument("--model", type=str, choices=['resnet', 'vit'], default='resnet', help="Model type: resnet or vit")
#     config.augment_argparse(parser)
#     config.collect_argparse_args(parser)
#     config.validate(mode='stderr')
#     config.summary()

#     mask = (np.random.rand(50_000) > 0.5)
#     loaders = make_dataloaders(mask=np.nonzero(mask)[0])
#     model = construct_model(model_type=config.model)
#     train(model, loaders)
#     res = evaluate(model, loaders)
#     print(mask.shape, res.shape)
#     return {
#         'masks': mask,
#         'margins': res
#     }
def main():
    config = get_current_config()
    parser = argparse.ArgumentParser(description='Fast CIFAR-10 training and comparison')
    parser.add_argument("--model", type=str, choices=['resnet', 'vit', 'both'], default='both',
                        help="Model type to train and evaluate: resnet, vit, or both")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    # Randomly mask 50% of the training data for subset-based evaluation
    mask = (np.random.rand(50_000) > 0.5)
    loaders = make_dataloaders(mask=np.nonzero(mask)[0])

    results = {}

    # Train and evaluate ResNet
    if config.model in ['resnet', 'both']:
        print("\nTraining ResNet...")
        resnet_model = construct_model(model_type='resnet')
        train(resnet_model, loaders)
        resnet_margins = evaluate(resnet_model, loaders)
        results['resnet'] = {
            'margins': resnet_margins,
            'average_margin': resnet_margins.mean()
        }
        print(f"ResNet Average Margin: {results['resnet']['average_margin']}")

    # Train and evaluate ViT
    if config.model in ['vit', 'both']:
        print("\nTraining ViT...")
        vit_model = construct_model(model_type='vit')
        train(vit_model, loaders)
        vit_margins = evaluate(vit_model, loaders)
        results['vit'] = {
            'margins': vit_margins,
            'average_margin': vit_margins.mean()
        }
        print(f"ViT Average Margin: {results['vit']['average_margin']}")

    # Print comparison results
    if config.model == 'both':
        print("\nModel Comparison Results:")
        print(f"ResNet Average Margin: {results['resnet']['average_margin']}")
        print(f"ViT Average Margin: {results['vit']['average_margin']}")
        print("Margin Difference (ViT - ResNet):",
              results['vit']['average_margin'] - results['resnet']['average_margin'])

    # Save masks and margins
    print("\nSaving results...")
    np.savez(f"{config.logdir}/comparison_results.npz", masks=mask, resnet_margins=results.get('resnet', {}).get('margins'),
             vit_margins=results.get('vit', {}).get('margins'))
    print("Results saved successfully!")

    # Return results for further usage
    return {
        'masks': mask,
        'results': results
    }
