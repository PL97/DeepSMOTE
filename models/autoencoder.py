from tabnanny import check
from turtle import forward
from typing import Counter
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10

import torchvision
import numpy as np
from pytorch_lightning.plugins import DDPPlugin

from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("../")
from utils.utils import UnNormalize, save_img_batch


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape[1:]) 



class res_encoder(nn.Module):
    '''
    resnet34_based encoder. The input to this model should be 3 channel and ideally 3x32x32. 
    To make the model fit the input, we remove two layers in the input and modify the first layer.
    input:
    pretrained: - use pretrained weights from ImageNet, default to be false and not recommend to
    turn to True as the structure are modified from the first conv layer
    '''
    def __init__(self, pretrained=False):
        super(res_encoder, self).__init__()

        self.base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv = nn.Sequential(
                self.base_model.bn1,
                self.base_model.relu,
                # self.base_model.maxpool,
                self.base_model.layer1,
                self.base_model.layer2,
                self.base_model.layer3,
                self.base_model.layer4,
                # nn.AdaptiveAvgPool2d((1,1))
                # nn.AvgPool2d(poolSize)
            )
        self.conv_random = nn.Sequential(
                nn.Conv2d(512, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),)
        self.fc = nn.Sequential(nn.Flatten(),
                nn.Linear(512, 512))
         
        
    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.conv(x)
        x = self.conv_random(x)
        return self.fc(x)
 

class encoder(nn.Module):
    '''
    encoder module contains two part, a set of conv layers and fc layers
    input:
    depth: -depth of the conv layer
    input_sample: -a sample of input
    hidden_dim: -hidden dimension
    '''
    def __init__(self, depth=3, 
                 input_sample=torch.zeros(1, 3, 32, 32), 
                 hidden_dim=512):
        super(encoder, self).__init__()
        
        self.conv = []
        for d in range(depth):
            self.conv.append(nn.Conv2d(input_sample.shape[1], 12, 4, stride=2, padding=1) if d == 0
                             else nn.Conv2d(12*(2**(d-1)), 12*(2**(d)), 4, stride=2, padding=1))
            self.conv.append(nn.LeakyReLU())
        
        self.conv = nn.Sequential(*self.conv)
        conv_dim = np.prod(self.conv(input_sample).shape[1:])
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_dim, hidden_dim),
            # nn.ReLU(),
        )
    
    def get_conv_output(self, x):
        return self.conv(x)
         
    def forward(self, x):
        x = self.conv(x)
        m = nn.Flatten()
        x = self.fc(x)
        return x
    
class decoder(nn.Module):
    '''
    decoder is ensembled similarly to encoder and has two component, deconv + fc
    input:
    depth: -depth of the deconv layer (convtrans2d to be more specifically)
    input_sample: -a sample of tensor output from the conv layer in encoder
    hidden_dim: -hidden dimension
    output_channel: -output channel which correspoding to the origional image channel
    '''
    def __init__(self, 
                 depth=3,
                 input_sample=torch.zeros(1, 48, 4, 4),
                 hidden_dim=512,
                 output_channel=3):
        
        super(decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, np.prod(input_sample.shape[1:])),
            # nn.ReLU()
            View(input_sample.shape)
        )
        self.deconv = []
        for d in range(depth):
            if d < depth - 1:
                self.deconv.append(nn.ConvTranspose2d((2**(depth-d-1))*12, (2**(depth-d-2))*12, 4, stride=2, padding=1))
                self.deconv.append(nn.LeakyReLU())
            else:
                self.deconv.append(nn.ConvTranspose2d((2**(depth-d-1))*12, output_channel, 4, stride=2, padding=1))
                self.deconv.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*self.deconv)

    def forward(self, x):
        x = self.fc(x)
        # x = x.view(input_sample.shape)
        return self.deconv(x)
    

class autoencoder(pl.LightningModule):
    '''
    Simple autoencoder compose of resnet-backended encoder and a three layer decoder
    input:
    depth: -depth of the conv layer in encoder and decoder
    input_sample: -a sample of input to the encoder
    hidden_dim: -the bottleneck output dimension (flattened tensor)
    '''
    def __init__(self,
                 depth=3,
                 input_sample=torch.zeros(1, 3, 32, 32), 
                 hidden_dim=512):
        super(autoencoder, self).__init__()
        self.encoder = encoder(depth=depth,
                               input_sample=input_sample, 
                                hidden_dim=hidden_dim)
        
        encoder_conv_output = self.encoder.get_conv_output(input_sample)
        
        self.decoder = decoder(depth=depth,
                                input_sample=encoder_conv_output,
                                hidden_dim=hidden_dim,
                                output_channel=input_sample.shape[1])
        # self.criterion = nn.BCELoss()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x.to(self.device))
        loss = self.criterion(x_hat, x)

        ## permute same class
        x_tilt=x.clone()
        y = y.detach().cpu().numpy()
        x_tilt_enc = self.encoder(x_tilt)
        idx = np.array(range(x_tilt.size(0)))
        for i in set(y):
            tmp = idx[y==i]
            np.random.shuffle(tmp)
            idx[y==i] = tmp

        x_tilt_enc_rand = x_tilt_enc[idx]
        x_tilt_enc_rand_dec = self.decoder(x_tilt_enc_rand)
        loss += 0.5*self.criterion(x_tilt_enc_rand_dec, x_tilt)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam([{"params": self.encoder.parameters()},
        {"params": self.decoder.parameters()}],
        lr=1e-3)

    def generate_using_smote(self, dl, model_series_number="version_0", save=False, saved_path="./data/"):
        features_list = []
        y_list = []
        for x, y in dl:
            features_list.extend(self.encoder(x).detach().cpu().numpy())
            y_list.extend(y.detach().cpu().numpy())
        
        print("Origional label distributin:", Counter(y_list))
        smt = SMOTE()
        features_sm, y_sm = smt.fit_resample(features_list, y_list)
        print("New label distributin:",Counter(y_sm))

        if save:
            total_imgs = len(features_sm)
            batch_size = 512
            for i in tqdm(range(0, total_imgs, batch_size)):
                tmp_x = torch.tensor(np.asarray(features_sm[i:i+batch_size])).to(self.device).float()
                tmp_y = y_sm[i:i+batch_size]
                recon_img = self.decoder(tmp_x)
                save_img_batch(recon_img, tmp_y, saved_path=saved_path, start_idx=i)
                

        else:
            x = torch.tensor(np.asarray(features_sm[:25])).to(self.device).float()
            recon_img = self.decoder(x)
            grid = torchvision.utils.make_grid(recon_img, nrow=5).permute(1, 2, 0)
            print(grid.shape)
            plt.figure(figsize=(50, 50))
            plt.imshow(grid.detach().cpu().numpy())
            plt.savefig(f"figures/{model_series_number}/synthetic.png")
    
    
    
if __name__ == "__main__":
    # pass
    # test case 2: create a valid autoencoder
    test_input = torch.zeros(512, 3, 32, 32)
    MyLightningModule = autoencoder(depth=5,
                    hidden_dim=1024,
                    input_sample=test_input)
    print(MyLightningModule)
    
    
    print(MyLightningModule(test_input).shape)
