from torchvision import transforms
from torchvision.datasets import CIFAR10
from pytorch_lightning.strategies.ddp import DDPStrategy
import pytorch_lightning as pl

import sys
sys.path.append("./")
from configs.config import parse_opts
from datasets.cifar100 import get_cifar100_dataset
from models.autoencoder import autoencoder
from utils.utils import show_reconstruct


if __name__ == "__main__":
    args = parse_opts()

    ## load dataset
    if args.dataset == "cifar100":
        dl = get_cifar100_dataset()

    ## define model
    model = autoencoder()

    ## train the model
    if args.train:
        trainer = pl.Trainer(max_epochs=300, 
                            accelerator="gpu", 
                            devices=4, 
                            strategy = DDPStrategy(find_unused_parameters=False),
                            log_every_n_steps=5)
        

        trainer.fit(model, train_dataloaders=dl)
    
    
    else:
        model_series_number = "version_1"
        MyLightningModule = autoencoder()
        model = MyLightningModule.load_from_checkpoint('lightning_logs/version_0/checkpoints/epoch=299-step=7500.ckpt')
        ## show reconstruction images and sythetic images
        x, y = next(iter(dl))
        show_reconstruct(x[:5], y[:5], model, model_series_number=model_series_number)
        # test case 3: generate new samples using SMOTE
        model.generate_using_smote(dl, model_series_number=model_series_number)