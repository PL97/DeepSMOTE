from torch.utils.data import sampler, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np

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

def get_cifar100_dataset():
    transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    
    cifar100_dataset = DataLoader(CIFAR100(root="./data", \
                                                train= True, \
                                                transform=transform, download=True), \
                                                batch_size=512, \
                                                num_workers=8, \
                                                persistent_workers=True, \
                                                sampler=DatasetSampler(idx=list(range(50000)), frac=1))
    return cifar100_dataset
