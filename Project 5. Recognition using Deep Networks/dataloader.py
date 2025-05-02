"""
1. Jyothi Vishnu Vardhan Kolla
2. Vidya Ganesh

Project-5: CS-5330 -> Spring 2023.

This file contains the code to load the
training and testing data into disk.
"""

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader


data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.1307, ), (0.3081,))
])


def create_dataloaders(batch_size: int):
    """
    Creates training and testing DataLoaders.

    Takes in a batch_size as parameter and return an
    train_dataloader, test_dataloader, class_names.

    Args:
     batch_size: Number of samples per batch in each of the DataLoaders.

    Returns:
     A tuple of (train_dataloader, test_dataloader, class_names).
     where class_names is a list of target_classes.
    """
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=data_transform,
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=data_transform
    )

    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, test_dataloader, class_names


class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


greek_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    GreekTransform(),
    torchvision.transforms.Normalize(
        (0.1307, ), (0.3081,))
])


def get_greekdata(batch_size: int):
    """
    Creates training and testing DataLoaders.

    Takes in a batch_size as parameter and return an
    train_dataloader, class_names.

    Args:
     batch_size: Number of samples per batch in each of the DataLoaders.

    Returns:
     A tuple of (train_dataloader, class_names).
     where class_names is a list of target_classes.
    """
    train_data = torchvision.datasets.ImageFolder("greek_train",
                                                  transform=greek_transform)
    class_names = train_data.classes

    # Prepare the data using dataLoader.
    greek_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    return greek_train, class_names


def get_FashionMnist(batch_size: int):
    """
    Creates training and testing DataLoaders.

    Takes in a batch_size as parameter and return an
    train_dataloader, test_dataloader, class_names.

    Args:
     batch_size: Number of samples per batch in each of the DataLoaders.

    Returns:
     A tuple of (train_dataloader, test_dataloader, class_names).
     where class_names is a list of target_classes.
    """
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=False)

    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False)

    class_names = train_data.classes

    return train_dataloader, test_dataloader, class_names