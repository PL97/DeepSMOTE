from torch.utils.data import sampler, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np

import sys
sys.path.append("../")
from datasets.shared import DatasetSampler

def get_cifar10_dataset():
    transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    
    cifar10_dataset = DataLoader(CIFAR10(root="./data", \
                                                train= True, \
                                                transform=transform, download=True), \
                                                batch_size=512, \
                                                num_workers=8, \
                                                persistent_workers=True, \
                                                sampler=DatasetSampler(idx=list(range(50000)), frac=1))
    return cifar10_dataset