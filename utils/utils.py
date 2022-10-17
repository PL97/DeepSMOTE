import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import os

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
    # unnorm = UnNormalize(fake=True)
    recon_img = model(x)
    grid1 = make_grid(x, nrow=1)
    grid2 = make_grid(recon_img, nrow=1)
    grid = torch.concat([grid1, grid2], axis=2).permute(1, 2, 0)
    plt.figure(figsize=(20, 50))
    plt.imshow(grid.detach().cpu().numpy())
    os.makedirs(f"figures/{model_series_number}/", exist_ok=True)
    plt.savefig(f"figures/{model_series_number}/recon_img_gallery.png")
    plt.close()