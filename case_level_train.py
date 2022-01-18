## Tools

import torch
import numpy as np
from PIL import ImageEnhance
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import models
from os.path import join
from tqdm import tqdm
import torch.optim as optim
import json,os, csv
import random
from collections import defaultdict
import argparse

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from tqdm.notebook import tqdm

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
parser.add_argument('--train_img', type = str, help='training image directory')
parser.add_argument('--jsonfile', type=str, help='train set json file')
parser.add_argument('--out_model', type=str, help='output model name')
args = parser.parse_args()

## Dataset & Dataloader
class Skull(Dataset):
    def __init__(self, img_root, json_file):
        """ Intialize the dataset """
        self.filenames = []
        self.img_root = img_root
        self.json_file = json_file
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.norm = lambda x : (x+1024)/4095*255
        self.profile = defaultdict(dict)
        self.files = []

        # read filenames
        file = open(json_file)
        j = json.load(file)

        ## Load by patients
        for patient in j["datainfo"]:
            series = patient[:-9]
            self.profile[patient] = j["datainfo"][patient]
        for i in self.profile:
            self.files.append(self.profile[i])
        file.close()

        self.len = len(self.files)

    def __getitem__(self, index):
        """ Get a sample from the dataset """

        ## Get item by patients
        f = self.files[index]
        data = np.load(join(self.img_root,f["path"]))
        data = self.norm(data)
        data = self.transform(data).float()
        coords = f["coords"]
        label = f["label"]
        label = int(label==1)
        series = f["series"]
        l = [data,label]

        return l

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
    
SkullSet = Skull(img_root = args.train_img, json_file=args.jsonfile)
train_set, val_set = torch.utils.data.random_split(SkullSet, [30000, 2665])
train_loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=0)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

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

## Save checkpoint
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

## Training
epochs = 10
lr = 3e-5
gamma = 0.7

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

mn_loss = 10
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)
    if epoch_val_loss < mn_loss:
        mn_loss = epoch_val_loss
        save_checkpoint(args.out_model,model,optimizer)
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )