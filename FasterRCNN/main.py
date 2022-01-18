import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pandas as pd
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
import numpy as np
import matplotlib.pyplot as plt
import os

from config import Config
from dataset import SkullDataset, SkullValDataset
from utils import split_data, get_pos_patient, convert
from worker import Worker
from validation.val import predict

def main(config:Config, device=torch.device("cuda")):
  img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(config.image_size)
  ])
  train_df, val_df = split_data(convert(config.train_json))
  train_dataset = SkullDataset(config, img_trans, train_df)
  val_dataset = SkullDataset(config, img_trans, val_df)
  dataloaders = {
    'train': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
  }
  net = torchvision.models.inception_v3(pretrained=True)
  net.AuxLogits = torch.nn.Identity()
  modules = list(net.children())[:-3]
  backbone = torch.nn.Sequential(*modules)
  backbone.out_channels = 2048

  anchor_generator = AnchorGenerator(sizes=config.anchor_sizes,aspect_ratios=config.anchor_aspect_ratios)
  roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=config.roi_output_size, sampling_ratio=2)
  model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
  worker = Worker(config, model,train_df, val_df, dataloaders, device)
  worker.box_predict()
  return worker

def predict_coordinate(config:Config, model, device=torch.device("cuda")):
  img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(config.image_size)
  ])

  test_df, csv_df = get_pos_patient(config)
  dataset = SkullValDataset(config, img_trans, test_df)
  dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
  predict( config, model, csv_df, dataloader, device)

if __name__  == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--train", help="Training data directory", required=True, type=str)
  parser.add_argument("-v", "--validation", help="test data directory", required=True, type=str)
  parser.add_argument("-j", "--json", help="train data json file", required=True, type=str)
  parser.add_argument("-o", "--output", help="Output file name", required=True, type=str)
  args = parser.parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  config = Config()
  config.init()
  config.output_csv_path = args.output
  config.train_dir = args.train
  config.test_dir = args.validation
  config.train_json = args.json
  w = main(args, device)
  # net = torchvision.models.resnet152(pretrained=True)
  # modules = list(net.children())[:-2]
  # backbone = torch.nn.Sequential(*modules)
  # backbone.out_channels = 2048

  # anchor_generator = AnchorGenerator(sizes=config.anchor_sizes,aspect_ratios=config.anchor_aspect_ratios)
  # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=config.roi_output_size, sampling_ratio=2)
  # model = torchvision.models.detection.FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
  # model.load_state_dict(torch.load("checkpoints\\resnet\epoch_12.pth"))
  predict_coordinate(config, w.model, device)