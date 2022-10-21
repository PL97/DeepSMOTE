import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import os
from joblib import Parallel, delayed
from skimage import io
import numpy as np

class UnNormalize(object):
    '''
    unnormalize the image according to the input mean and std
    defult are imagenet mean and std
    '''
    def __init__(self, fake=False, norm_type="others", grayscale=False):
        if grayscale:
            mean = [0.5]
            std = [0.5]
        else:
            if norm_type == "imagenet":
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
            else:
                mean = (0.5, 0.5, 0.5)
                std = (0.5, 0.5, 0.5)
        self.mean = mean
        self.std = std
        self.fake = fake

    def __call__(self, tensor, cwh=True, batch=False):
        if self.fake:
            return tensor
            
        for idx, (m, s) in enumerate(zip(self.mean, self.std)):
            if cwh:
                if batch:
                    tensor[:, idx, :, :] = tensor[:, idx, :, :]*s + m
                else:
                    tensor[idx, :, :] = tensor[idx, :, :]*s + m
            else:
                tensor[:, :, idx] = tensor[:, :, idx]*s + m
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def show_reconstruct(x, y, model, model_series_number="version_0"):
    
    plt.figure()
    unnorm = UnNormalize(norm_type="imagenet")
    recon_img = unnorm(model(x))
    grid1 = make_grid(x, nrow=1)
    grid2 = make_grid(recon_img, nrow=1)
    grid = torch.concat([grid1, grid2], axis=2).permute(1, 2, 0)
    plt.figure(figsize=(20, 50))
    plt.imshow(grid.detach().cpu().numpy())
    os.makedirs(f"figures/{model_series_number}/", exist_ok=True)
    plt.savefig(f"figures/{model_series_number}/recon_img_gallery.png")
    plt.close()


def save_img(img, l, idx, saved_path, from_tensor=True):
    '''
    save image to the specified path
    input:
    img: - image to be saved
    l: label of the image
    idx: unique interage to generated file name


    saved path would be {$saved_path}/{$l}/{idx}.png
    '''
    if from_tensor:
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        img = (img*256).astype(np.uint8)
    io.imsave(f"{saved_path}/{l}/{idx}.png", img)



def save_img_batch(imglist, labellist, saved_path, start_idx):
    ## create saved folder if not exits
    unnorm = UnNormalize(norm_type="imagenet")
    for l in set(labellist):
        os.makedirs(f"{saved_path}/{l}", exist_ok=True)
    idx_list = list(range(start_idx, len(imglist)+start_idx))
    Parallel(n_jobs=10)(delayed(unnorm(save_img))(img, l, idx, saved_path)
                        for img, l, idx in 
                        zip(imglist, labellist, idx_list))