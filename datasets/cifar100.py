from torch.utils.data import sampler, DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import torch
from PIL import Image
from typing import Any, Callable, Optional, Tuple

class DatasetSampler(sampler.Sampler):
    '''
    A simple sampler that random sample image from a dataloader
    input:
    frac: - percentage of files to use, [0, 1]
    '''
    def __init__(self, frac, idx):
        self.idx = idx[:int(len(idx)*frac)]

    def __iter__(self):
        for i in np.random.permutation(self.idx):
                yield i

    def __len__(self):
        return len(self.idx)
    

class My_CIFAR100(CIFAR100):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        ## change the label to binary
        target = int(target>=99)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target




def get_cifar100_dataset():
    transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    
    cifar100_dataset = My_CIFAR100(root="./data", \
                                                train= True, \
                                                transform=transform, download=True)
                
    cifar100_dataloader = DataLoader(cifar100_dataset,\
                                        batch_size=512, \
                                        num_workers=8, \
                                        persistent_workers=True, \
                                        sampler=DatasetSampler(idx=list(range(50000)), frac=1))
    return cifar100_dataloader
