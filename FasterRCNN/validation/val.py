import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import pandas as pd
from config import Config
from validation.result import Result
from utils import plot_boxes, gen_target, extract_coords, cal_F1, flatten_coords

@torch.no_grad()
def test_model(model:nn.Module, df, dataloader:DataLoader, device=torch.device("cuda")):
  model.eval()
  model = model.to(device)
  res = Result(df)
  for imgs, labels, ind in dataloader:
    imgs = imgs.to(device)
    neg = labels == -1
    labels[neg] = 0
    labels = labels.to(device)

    output= model(imgs)
    pred = (output>0.5).long().squeeze()
    labels[neg] = -1
    ind = ind.tolist()
    res.add(ind, pred.cpu().tolist(), labels.cpu().tolist(), df.loc[ind, 'coords'].to_list(), df.loc[ind, 'coords'].to_list())

  res.assign_label()
  acc = res.cal_acc()
  res.to_csv('test.csv')
  # print("Val Accuracy: [{}/{}] {:.2f}%".format(corrects, len(dataloader.dataset), 100*corrects/len(dataloader.dataset)))
  return acc

@torch.no_grad()
def predict_coord(config, model, df, dataloader:DataLoader, device=torch.device("cuda")):
  model.eval()
  model = model.to(device)
  sampled = random.sample(range(len(dataloader)-2), 5)
  predicts = []
  ground_truths = []
  for i, (imgs, _, ids) in enumerate(dataloader):
    imgs = imgs.to(device)
    output = model(imgs)
    gt_coords = df.loc[ids, 'coords']
    predicts.extend(flatten_coords(extract_coords(output, config.box_score_threshold)))
    ground_truths.extend(flatten_coords(gt_coords.to_list()))
    if i in sampled:
      pred_boxes =  [output[0]["boxes"][i].tolist() for i in range(len(output[0]["boxes"])) if output[0]["scores"][i] >= config.box_score_threshold]
      print(output)
      print("predict coords:", extract_coords(output, config.box_score_threshold))
      print("ground truth coords:", gt_coords.to_list())
      print("pred_boxes:", pred_boxes)
      print("ground truth", gen_target(gt_coords))
      print()
      plot_boxes(imgs, output, gt_coords, config.box_score_threshold)
  TP, FN, FP, F1_score = cal_F1(predicts, ground_truths, '')
  print(f"TP: {TP}, FN: {FN}, FP: {FP}, F1 score: {F1_score}")
  return F1_score
    
    

@torch.inference_mode()
def predict(config:Config, model:nn.Module, output_df, dataloader: DataLoader, device=torch.device("cuda")):
  output_df = output_df.set_index("id")
  model = model.to(device)
  model.eval()
  total = 0
  for imgs, ids in dataloader:
    imgs = imgs.to(device)
    output = model(imgs)
    pred_coords = flatten_coords(extract_coords(output, config.box_score_threshold))
    str_coords = list()
    new_labels = list()
    for x in pred_coords:
      temp = [str(int(y)) for y in x]
      c = ' '.join(temp)
      if c == '':
        str_coords.append('')
        new_labels.append(-1)
      else:
        str_coords.append(c)
        new_labels.append(1)
      total+=1
    output_df.loc[ids, "coords"] = str_coords
    output_df.loc[ids, "label"] = new_labels
  output_df.to_csv(config.output_csv_path)
  return output_df


  
  