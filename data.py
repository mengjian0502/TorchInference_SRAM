"""
Dataset
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import datasets, transforms

def get_loader(batch_size, data_path):
    transform = transforms.ToTensor()
    trainset = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    testset = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    return trainloader, testloader

