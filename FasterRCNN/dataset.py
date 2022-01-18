import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import json
import os
from config import Config
from typing import Dict, List, Tuple

class SkullDataset(Dataset):
  def __init__(self, args:Config, transform, data_df:pd.DataFrame) -> None:
    super(SkullDataset, self).__init__()
    self.data_df = data_df
    self.image_dir:str = args.train_dir
    self.transform = transform
    self.len:int = len(self.data_df)

  def __getitem__(self, index:int) -> Tuple[torch.Tensor, int, List[List[int]]]:
    path = os.path.join(self.image_dir, self.data_df.loc[index, 'path'])
    label = self.data_df.loc[index, 'label']
    image = np.load(path).astype(float)
    lb, up = image>=0, image<2550
    bound = np.logical_and(lb, up)
    image[bound] = image[bound] / 2550
    image[image<0] = 0
    image[image>=2550] = 1
    image = self.transform(image)
    return image.float(), label, index

  def __len__(self) -> int:
    return self.len

class SkullValDataset(Dataset):
  def __init__(self, args:Config, transform, data_df:pd.DataFrame) -> None:
    super(SkullValDataset, self).__init__()
    self.data_df = data_df
    self.image_dir:str = args.test_dir
    self.transform = transform
    self.len:int = len(self.data_df)

  def __getitem__(self, index:int) -> Tuple[torch.Tensor, int, List[List[int]]]:
    path = os.path.join(self.image_dir, self.data_df.loc[index, 'path'])
    image = np.load(path).astype(float)
    image_id = self.data_df.loc[index, 'idx']
    lb, up = image>=0, image<2550
    bound = np.logical_and(lb, up)
    image[bound] = image[bound] / 2550
    image[image<0] = 0
    image[image>=2550] = 1
    image = self.transform(image)
    return image.float(), image_id

  def __len__(self) -> int:
    return self.len

class SkullSampler(Sampler):
  def __init__(self, data_source) -> None:
    self.data = data_source


if __name__ == "__main__":
  from utils import split_data
  from torch.utils.data import DataLoader
  args = Config()
  trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=0.5, std=0.5)])
  df = pd.read_json(args.test_data_json, orient='index')
  dataset = SkullValDataset(args, trans, df)
  print(df)

  dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
  dataiter = iter(dataloader)
  imgs, labels = next(dataiter)
  print(labels)
  plt.imshow(imgs[0][0])
  plt.show()
  val = split_data(args.train_json)[1]
  print(val.loc[[0, 1, 2], "coords"].tolist())
  for imgs, labels in dataloader:
    print(labels)
    break
  # idx = 0
  # while idx < len(dataset) and len(dataset[idx][2])<5:
  #   idx += 1
  # print(idx)
  # print(dataset[idx])
  # print(dataset[idx][0][0].shape)
  # print(torch.max(dataset[idx][0][0]), torch.min(dataset[idx][0][0]))
  # plt.imshow(dataset[idx][0][0], cmap='gray')
  # plt.show()
  # with open(args.train_json, 'r')
