from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning.strategies.ddp import DDPStrategy
import pytorch_lightning as pl
import torch
import os

import sys
sys.path.append("./")
from configs.config import parse_opts
from datasets.cifar100 import get_cifar100_dataset
from datasets.cifar10 import get_cifar10_dataset
from datasets.HAM10000 import get_HAM_dataloader
from models.autoencoder import AE, AE_LM
from utils.utils import show_reconstruct


if __name__ == "__main__":
    args = parse_opts()

    #saved_name = "epoch=299-step=29400.ckpt"
    # saved_name = "epoch=299-step=3900.ckpt"
    # saved_name = "epoch=0-step=5.ckpt"
    model_series = f"version_{args.model_series}"
    saved_name = os.listdir(f'lightning_logs/{model_series}/checkpoints/')[0]

    ## load dataset and define the model
    if args.dataset == "cifar100":
        dl = get_cifar100_dataset()
        test_input = torch.zeros(1, 3, 32, 32)
        model = AE(depth=3,
                      hidden_dim=512,
                      input_sample=test_input)
        MyLightningModule = AE_LM(AE=model)

    elif args.dataset == "cifar10":
        dl = get_cifar10_dataset()
        test_input = torch.zeros(1, 3, 32, 32)
        model = AE(depth=3,
                      hidden_dim=512,
                      input_sample=test_input)
        MyLightningModule = AE_LM(AE=model)
        
    elif args.dataset == "HAM10000":
        dl = get_HAM_dataloader()
        test_input = torch.zeros(1, 3, 224, 224)
        model = AE(depth=1,
                        hidden_dim=1024,
                        input_sample=test_input)
        MyLightningModule = AE_LM(AE=model)
        


    if args.synthesizing:
        
        MyLightningModule = MyLightningModule.load_from_checkpoint(f'lightning_logs/{model_series}/checkpoints/{saved_name}', AE=AE)
        MyLightningModule.eval()
        MyLightningModule.generate_using_smote(dl, model_series_number=model_series, save=True, saved_path=f"./data/{args.dataset}")
        exit("finished")


    ## train the model
    if args.train:
        trainer = pl.Trainer(max_epochs=300, 
                            accelerator="gpu", 
                            devices=1, 
                            strategy = DDPStrategy(find_unused_parameters=False),
                            log_every_n_steps=20)
        

        trainer.fit(MyLightningModule, train_dataloaders=dl)
    
    
    else:
        model_series = f"version_{args.model_series}"
        MyLightningModule = AE_LM.load_from_checkpoint(f'lightning_logs/{model_series}/checkpoints/{saved_name}', AE=model)
        MyLightningModule.eval()
        ## show reconstruction images and sythetic images
        x, y = next(iter(dl))
        show_reconstruct(x[:5], y[:5], MyLightningModule, model_series_number=model_series)
        # test case 3: generate new samples using SMOTE
        MyLightningModule.generate_using_smote(dl, model_series_number=model_series)
