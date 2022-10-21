from torch.utils.data import sampler, DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import torch

import sys
sys.path.append("../")
from datasets.shared import DatasetSampler

def get_cifar10_dataset():
    transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    
    cifar10_dataset = CIFAR10(root="./data", \
                                train= True, \
                                transform=transform, \
                                download=True)
    targets = cifar10_dataset.targets
    targets = torch.IntTensor((np.asarray(targets)>=9).astype(int))
    inputs = torch.tensor(cifar10_dataset.data.transpose(0, 3, 1, 2)).float()
    
    cifar10_dataset = TensorDataset(inputs, targets)
                
    cifar10_dataloader = DataLoader(cifar10_dataset,\
                                        batch_size=512, \
                                        num_workers=8, \
                                        persistent_workers=True, \
                                        sampler=DatasetSampler(idx=list(range(50000)), frac=1))
    
    
    return cifar10_dataloader