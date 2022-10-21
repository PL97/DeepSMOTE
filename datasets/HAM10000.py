from PIL import Image
import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torch
import os

import sys
sys.path.append("../")
from datasets.shared import DatasetSampler

lesion_type_dict = {'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'}

lesion_to_num = {'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6}


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result



class HAM(Dataset):
    def __init__(self, df, root_dir, mode="train"):
        assert mode in ['train', 'val'], "invalid mode"
        self.path = list(df['image_id'])
        # self.label = pd.Categorical(df["dx"]).codes
        self.label = [lesion_to_num[x] for x in df['dx']]
        self.root_dir = root_dir
        self.mode = mode

        self.transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop((224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}

    def __getitem__(self, index):
        p = self.root_dir + self.path[index] + ".jpg"
        img = Image.open(p).convert("RGB")
        img = expand2square(img, (255, 255, 255))
        # img = padAndResize(img)
        label = self.label[index]
        # label = torch.FloatTensor(label)
        img = self.transform[self.mode](img)
        return img, label

    def __len__(self):
        return len(self.path)

        


def get_HAM_dataloader(mode="train"):
    run_number = 1
    # if ctx['server'] == "MSI":
    root_path = "/home/jusun/shared/HAM/"
    # else:
    #root_path = "/home/le/TL/sync/truncatedTL/HAM/"
    df = pd.read_csv(os.path.join(root_path, f"{run_number}/{mode}.csv"))
    ham_ds = HAM(df=df, root_dir=root_path+"jpgs/")
    dataloader = torch.utils.data.DataLoader(ham_ds, \
                                            batch_size=128, \
                                            # shuffle=True, \
                                            num_workers=24, \
                                            persistent_workers=True, \
                                            sampler=DatasetSampler(idx=list(range(df.shape[0])), frac=0.1))


    return dataloader


if __name__ == "__main__":
    # dl = get_HAM_dataloader()

    
    # for x, y in dl:
    #     print(x.shape, y.shape)
    #     asdf
    
    pass
