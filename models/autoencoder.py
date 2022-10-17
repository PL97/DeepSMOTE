from tabnanny import check
from turtle import forward
from typing import Counter
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision
import numpy as np
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies.ddp import DDPStrategy
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from utils.utils import UnNormalize


class DatasetSampler(torch.utils.data.sampler.Sampler):
        def __init__(self, frac, idx):
            self.idx = idx[:int(len(idx)*frac)]

        def __iter__(self):
            # return iter(self.idx)
            for i in np.random.permutation(self.idx):
                    yield i

        def __len__(self):
            return len(self.idx)


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
    def __init__(self):
        super(encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
        )   
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48*4*4, 512),
            # nn.ReLU(),
        )
         
         
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    
class decoder(nn.Module):
    '''
    Three layer decoder purely based on convtranspose2d
    '''
    def __init__(self):
        super(decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 48*4*4),
            # nn.ReLU()
        )
        self.deconv = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 48, 4, 4)
        return self.deconv(x)
    

class autoencoder(pl.LightningModule):
    '''
    Simple autoencoder compose of resnet-backended encoder and a three layer decoder
    '''
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = encoder()
        
        self.decoder = decoder()
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
        loss += self.criterion(x_tilt_enc_rand_dec, x_tilt)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam([{"params": self.encoder.parameters()},
        {"params": self.decoder.parameters()}],
        lr=1e-3)

    def generate_using_smote(self, dl):
        features_list = []
        y_list = []
        for x, y in dl:
            y = y>=9
            features_list.extend(self.encoder(x).detach().cpu().numpy())
            y_list.extend(y.detach().cpu().numpy())
        
        print(Counter(y_list))
        smt = SMOTE()
        features_sm, y_sm = smt.fit_resample(features_list, y_list)
        print(Counter(y_sm))


        x = torch.tensor(features_list[:25]).to(self.device)
        recon_img = self.decoder(x)
        grid = torchvision.utils.make_grid(recon_img, nrow=5).permute(1, 2, 0)
        print(grid.shape)
        plt.figure(figsize=(50, 50))
        plt.imshow(grid.detach().cpu().numpy())
        plt.savefig("synthetic.png")



def show_reconstruct(x, y, model):
    
    plt.figure()
    unnorm = UnNormalize(fake=True)
    for i in range(x.shape[0]):
        input_x = x[i].unsqueeze(0)
        x_hat = model(input_x)

        plt.subplot(2, 5, 1+i%5)
        plt.imshow(unnorm(input_x.squeeze()).detach().cpu().numpy().transpose(1, 2, 0))
        plt.subplot(2, 5, 1+i%5+5)
        plt.imshow(unnorm(x_hat.squeeze()).detach().cpu().numpy().transpose(1, 2, 0))
    
    plt.savefig("recon_img_gallery.png")
    
    
    
    
if __name__ == "__main__":
    
    # ## test case 1: train a autoencoder
    # # mean = (0.485, 0.456, 0.406)
    # # std = (0.229, 0.224, 0.225)
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)
    # transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         # transforms.RandomHorizontalFlip(),
    #         # transforms.RandomVerticalFlip(),
    #         # transforms.Normalize(mean, std),
    # ])
    
    # mnist_dataset = torch.utils.data.DataLoader(CIFAR10(root="/home/le/DeepSmote/data", train= True, transform=transform, download=True),
    #                                             batch_size=512,
    #                                             num_workers=8,
    #                                             persistent_workers=True,
    #                                             shuffle=True)
    #                                             # sampler=DatasetSampler(idx=list(range(50000)), frac=0.01))

    
    
    # trainer = pl.Trainer(max_epochs=300, 
    #                      accelerator="gpu", 
    #                      devices=4, 
    #                      strategy = DDPStrategy(find_unused_parameters=False),
    #                      log_every_n_steps=5)
    # model = autoencoder()

    # trainer.fit(model, train_dataloaders=mnist_dataset)

    # x, y = next(iter(mnist_dataset))
    # show_reconstruct(x[:5], y[:5], model)
    
    # # test case 2: create a valid autoencoder
    # net = autoencoder()
    # test_input = torch.zeros(1, 3, 32, 32)
    
    # print(net(test_input).shape)

    # test case 3: generate new samples using SMOTE
    transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    mnist_dataset = torch.utils.data.DataLoader(CIFAR10(root="/home/le/DeepSmote/data", train= True, transform=transform, download=True),
                                                batch_size=512,
                                                num_workers=8,
                                                persistent_workers=True,
                                                # shuffle=True)
                                                sampler=DatasetSampler(idx=list(range(50000)), frac=0.01))
    MyLightningModule = autoencoder()
    model = MyLightningModule.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=299-step=7500.ckpt')
    model.generate_using_smote(mnist_dataset)

