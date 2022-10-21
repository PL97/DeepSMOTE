from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning.strategies.ddp import DDPStrategy
import pytorch_lightning as pl
import torch

import sys
sys.path.append("./")
from configs.config import parse_opts
from datasets.cifar100 import get_cifar100_dataset
from datasets.cifar10 import get_cifar10_dataset
from datasets.HAM10000 import get_HAM_dataloader
from models.autoencoder import autoencoder
from utils.utils import show_reconstruct


if __name__ == "__main__":
    args = parse_opts()

    ## load dataset and define the model
    if args.dataset == "cifar100":
        dl = get_cifar100_dataset()
        test_input = torch.zeros(1, 3, 32, 32)
        MyLightningModule = autoencoder(depth=4,
                      hidden_dim=512,
                      input_sample=test_input)

    elif args.dataset == "cifar10":
        dl = get_cifar10_dataset()
        test_input = torch.zeros(1, 3, 32, 32)
        MyLightningModule = autoencoder(depth=4,
                      hidden_dim=512,
                      input_sample=test_input)
        
    elif args.dataset == "HAM10000":
        dl = get_HAM_dataloader()
        test_input = torch.zeros(1, 3, 224, 224)
        MyLightningModule = autoencoder(depth=5,
                      hidden_dim=1024,
                      input_sample=test_input)

    ## define model
    model = autoencoder()

    if args.synthesizing:
        model_series = f"version_{args.model_series}"
        model = MyLightningModule.load_from_checkpoint(f'lightning_logs/{model_series}/checkpoints/epoch=299-step=7500.ckpt')
        model.generate_using_smote(dl, model_series_number=model_series, save=True, saved_path="./data/cifar100")
        exit("finished")


    ## train the model
    if args.train:
        trainer = pl.Trainer(max_epochs=300, 
                            accelerator="gpu", 
                            devices=4, 
                            strategy = DDPStrategy(find_unused_parameters=False),
                            log_every_n_steps=5)
        

        trainer.fit(model, train_dataloaders=dl)
    
    
    else:
        model_series = f"version_{args.model_series}"
        model = MyLightningModule.load_from_checkpoint(f'lightning_logs/{model_series}/checkpoints/epoch=299-step=7500.ckpt')
        ## show reconstruction images and sythetic images
        x, y = next(iter(dl))
        show_reconstruct(x[:5], y[:5], model, model_series_number=model_series)
        # test case 3: generate new samples using SMOTE
        model.generate_using_smote(dl, model_series_number=model_series)
