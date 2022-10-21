from torch.utils.data import sampler, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
from PIL import Image
from typing import Any, Callable, Optional, Tuple

import sys
sys.path.append("../")
from datasets.shared import DatasetSampler

class My_CIFAR10(CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        
        ## change the label to binary
        target = int(target>=9)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        

def get_cifar10_dataset():
        transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        cifar10_dataset = My_CIFAR10(root="./data", \
                                        train= True, \
                                        transform=transform, download=True)
        cifar10_dataloader = DataLoader(cifar10_dataset, \
                        batch_size=512, \
                        num_workers=8, \
                        persistent_workers=True, \
                        sampler=DatasetSampler(idx=list(range(50000)), frac=1))
        return cifar10_dataloader