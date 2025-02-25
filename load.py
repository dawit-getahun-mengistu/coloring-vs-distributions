import os
import random
import numpy as np

import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

import cfg


directory = cfg.DATASET_PATH

img_size = cfg.IMG_SIZE


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


class MNIST:
    def __init__(self, data_path: str = directory, img_size: int = img_size, train_batch_size: int = cfg.TRAIN_BATCH_SIZE, test_batch_size: int = cfg.TEST_BATCH_SIZE):
        self.data_path = os.makedirs(
            os.path.join(data_path, 'mnist'), exist_ok=True)
        self.img_size = img_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.transform = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

        self.train_dataset = datasets.MNIST(
            root=self.data_path, train=True, transform=self.transform, download=True
        )
        self.test_dataset = datasets.MNIST(
            root=self.data_path, train=False, transform=self.transform, download=True
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )


class FashionMNIST:
    def __init__(self, data_path: str = directory, img_size: int = img_size, train_batch_size: int = cfg.TRAIN_BATCH_SIZE, test_batch_size: int = cfg.TEST_BATCH_SIZE):
        self.data_path = os.makedirs(
            os.path.join(data_path, 'fashion-mnist'), exist_ok=True)
        self.img_size = img_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.transform = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

        self.train_dataset = datasets.MNIST(
            root=self.data_path, train=True, transform=self.transform, download=True
        )
        self.test_dataset = datasets.MNIST(
            root=self.data_path, train=False, transform=self.transform, download=True
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False
        )
