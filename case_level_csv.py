## Tools

import torch
import numpy as np
from PIL import ImageEnhance
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import models
from os.path import join
import torch.optim as optim
import json,os, csv
import random
from collections import defaultdict
import argparse

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

## Device
device = 'cuda'

## Random Seed
seed = 168
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

## Arg Parser
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
parser.add_argument('--model', type = str, help='load model')
parser.add_argument('--testdir', type=str, help='test image directory')
parser.add_argument('--out_csv', type=str, help='output csv file')
args = parser.parse_args()

## Model
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    
from linformer import Linformer
efficient_transformer = Linformer(
    dim=128,
    seq_len=1024+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)
model = ViT(
    dim=128,
    image_size=512,
    patch_size=16,
    num_classes=2,
    transformer=efficient_transformer,
    channels=1,
).to(device)

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
optimizer = optim.Adam(model.parameters(), lr=3e-5)

load_checkpoint(args.model,model,optimizer)

def write_csv(test_dir,model,out_csv):
    trans = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Resize(600),
                ])
    norm = lambda x : (x+1024)/4095*255
    series = sorted(os.listdir(test_dir))
    book = dict()
    with open(out_csv, 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','label','coords'])
        for s in series:
            series_name = join(test_dir,s)
            series_fn = []
            series_list = np.array([])
            for p in sorted(os.listdir(series_name)):
                series_fn.append(p)
                patient_name = join(series_name,p)
                data = np.load(patient_name)
                data = norm(data)
                data = trans(data).float().unsqueeze(0)
                data = data.to(device)
                label = model(data)
                series_list = np.append(series_list,label.argmax(dim=1).item())
            if any(series_list):
                series_list[series_list==0]=-1
            for i in range(len(series_list)):
                if int(series_list[i])==1:
                    writer.writerow([series_fn[i][:-4],int(series_list[i]),"256 256"])
                else:
                    writer.writerow([series_fn[i][:-4],int(series_list[i]),None])
                    
write_csv(args.testdir,model,args.out_csv)