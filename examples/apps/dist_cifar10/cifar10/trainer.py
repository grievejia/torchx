import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from typing import Optional, Dict, List
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import argparse

PATH = "/home/ubuntu/data"

BATCH_SIZE = 64


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


def create_data_module():
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ])

    return CIFAR10DataModule(
        data_dir=PATH,
        batch_size=BATCH_SIZE,
        #     num_workers=NUM_WORKERS,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            'scheduler': OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch lightning + classy vision TorchX example app"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs to train"
    )
    parser.add_argument(
        "--ngpus", type=int, default=0, help="num gpus"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="data dir",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        help="path to place the tensorboard logs",
        default="/tmp",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    PATH = args.data_dir
    cifar10_dm = create_data_module()

    model = LitResnet(lr=0.05)
    model.datamodule = cifar10_dm

    Trainer(gpus=args.ngpus).fit(model, cifar10_dm)
