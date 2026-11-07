from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

transform = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
# transform = transforms.Compose([
#     transforms.Lambda(lambda img: img.convert('RGB')),  
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # transforms.Normalize(
#     #     mean=[0.485, 0.456, 0.406],
#     #     std=[0.229, 0.224, 0.225]
#     # ),
# ])

class NSD_Dataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if (f.endswith(".bmp") and not f.startswith("MFO"))])
        # MFO stim are non objects (non-coco)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def get_NSD_dataset_imgs(folder, batch_size=100, num_workers=4):
    dataset = NSD_Dataset(folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader




def get_tripleN_fr(folder, category_selective=False, n_ch_samples=None, seed=0):
    triple_n = pd.read_csv(folder) #tripleN unit FRs
    triple_n = triple_n.sort_values(by="Reliability", ascending=False) #sort by reliability
    start_ind = int(np.where(["stim_1" == s for s in list(triple_n.columns)])[0][0])
    triple_n_FR = triple_n.iloc[:,start_ind:start_ind+1000].to_numpy() # only first 1000 stim are object stim from coco
    triple_n_FR = np.transpose(triple_n_FR)   # n_stim x n_channels
    

    if category_selective:
        cats = ['O', 'B', 'F']
        cat_units_FR = {}

        for cat in cats:
            cat_units_indices = np.where(triple_n['Category'].values == cat)[0]
            cat_FR = triple_n_FR[:, cat_units_indices]
            cat_units_FR[cat] = cat_FR
        return cat_units_FR

    else:
        if n_ch_samples:
            # randomly sample
            rs = np.random.RandomState(seed)
            sample_indices  = rs.choice(triple_n_FR.shape[1], size=n_ch_samples, replace=False)
            triple_n_FR = triple_n_FR[:, sample_indices]

        return triple_n_FR

def get_tripleN_reliability(folder):
    triple_n = pd.read_csv(folder) #tripleN unit FRs
    triple_n = triple_n.sort_values(by="Reliability", ascending=False)
    neuron_reliability = list(triple_n['Reliability'])
    return neuron_reliability

