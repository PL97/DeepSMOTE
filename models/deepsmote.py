from sklearn.neighbors import NearestNeighbors
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import sys
sys.append("../")
from models.autoencoder import autoencoder



# class LitModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.l1 = nn.Linear(28 * 28, 10)

#     def forward(self, x):
#         return torch.relu(self.l1(x.view(x.size(0), -1)))


#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.02)


def G_SM1(X, y,n_to_sample,cl):
    
    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample