from torch.utils.data import sampler
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