from typing import Dict, List
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
import json
import os
import numpy as np
from sklearn.cluster import DBSCAN

from config import Config

def gen_bbox_label(feat_stride, x, y, coords):

  bbox = (x*feat_stride, y*feat_stride, (x+1)*feat_stride, (y+1)*feat_stride)
  candidate = list()
  for gx, gy in coords:
    if gx>=bbox[0] and gx<=bbox[2] and gy>= bbox[1] and gy<=bbox[3]:
      candidate.append((gx, gy))
  return

def convert(original_path):
  with open(original_path, 'r') as fin:
    data:Dict = json.load(fin)['datainfo']

  patients = dict()
  for i, (key, item) in enumerate(data.items()):
    item["idx"] = key
    patients[str(i)] = item
  
  positive_pat_df = pd.DataFrame.from_dict(patients, orient='index')
  positive_pat_df = positive_pat_df.loc[positive_pat_df["label"]==1].reset_index().drop(columns=["index"])
  return positive_pat_df

def split_data(df:pd.DataFrame):
  patients = df.groupby('series')
  train_sample, test_sample = train_test_split(patients.size(), test_size=0.1)
  train = df.set_index('series').loc[train_sample.index].sort_values(by='idx').reset_index()
  test = df.set_index('series').loc[test_sample.index].sort_values(by='idx').reset_index()
  return train, test

def balance(json_path:str):
  df = pd.read_json(json_path, orient='index')
  patients = df.groupby('series')
  train_sample, test_sample = train_test_split(patients.size(), test_size=0.2)
  train = df.set_index('series').loc[train_sample.index]
  test = df.set_index('series').loc[test_sample.index].sort_values(by='idx').reset_index()
  print("All")
  print(df.groupby('label').size())
  print("Validation Set")
  print('test', test.groupby('label').size())
  print("Train Set")
  print(train.groupby('label').size())

  repeat = list()
  for s in train_sample.index:
    patient = train.loc[s]
    label = patient.groupby('label').size()
    if len(label) == 1:
      repeat.append(1)
    else:
      repeat.append(round(label[1]/(label[1]+label[-1])*10)*2)
  train["repeat"] = [0] * len(train)
  for s, r in zip(train_sample.index, repeat):
    train.loc[s, 'repeat'] = r
  repeat_train = train.sort_values(by='idx').reset_index()
  train = repeat_train.loc[repeat_train.index.repeat(repeat_train.repeat)].reset_index(drop=True)
  print(train.groupby('label').size())
  
  return train, test

def extract_bbox(coords: List[List[int]]):
  box = list()
  for c in coords:
    box.append([c[0]-16, c[1]-16, c[0]+16, c[1]+16])
  return torch.FloatTensor(box)

def gen_target(coords:List[List[List[int]]], image_size=512, device=torch.device("cuda")):
  targets = list()
  for c in coords:
    s = len(c)
    label = [1]*s
    target = {}
    
    target["boxes"] = extract_bbox(c).to(device)
    target["labels"] = torch.tensor(label).to(device)
  targets.append(target)
  return targets

def plot_boxes(imgs:torch.Tensor, target, gt_coords:pd.Series, thres=0.5):
  imgs = imgs.cpu().numpy()
  gt_coords = gt_coords.to_list()
  for i in range(len(imgs)):
    plt.imshow(imgs[0][i])
    ax = plt.gca()
    for j, box in enumerate(target[i]["boxes"]):
      if target[i]["labels"][j] != 1 or target[i]["scores"][j] < thres:
        continue
      pred_rect = patches.Rectangle((box[0].item(),box[1].item()), (box[2]-box[0]).item(), (box[3]-box[1]).item(),linewidth=2,edgecolor='cyan',fill = False)
      ax.add_patch(pred_rect)
    for j in range(len(gt_coords[i])):
      gt_rect = patches.Rectangle((gt_coords[i][j][0]-16 ,gt_coords[i][j][1]-16), 32, 32 ,linewidth=2,edgecolor='red',fill = False)
      ax.add_patch(gt_rect)
    plt.show()
    plt.clf()


def extract_coords(targets:List[Dict[str, torch.Tensor]], thres=0.5):
  coords = list()
  for i in range(len(targets)):
    c = list()
    for j, box in enumerate(targets[i]["boxes"]):
      if targets[i]["labels"][j] != 1 or targets[i]["scores"][j] < thres:
        continue
      c.append([(box[0].item()+box[2].item())/2, (box[1].item()+box[3].item())/2])
    coords.append(c)
  return coords

def flatten_coords(coords: List[List[int]]):
  new_coords = list()
  for c in coords:
    c.sort()
    flatten = list()
    for box in c:
      flatten.extend(box)
    new_coords.append(flatten)
  return new_coords

def cal_TPFNFP(preds, gts, r=32):
	# preds = [x1, y1, x2, y2, x3, y3, ...], gts = [x1, y1, x2, y2, ...]
	# Manhattan distance = |x1 - x2| + |y1 - y2|
	assert len(preds) % 2 == 0, f"length of pred is wrong : {len(preds)}"
	assert len(gts) % 2 == 0, f"length of gt is wrong: {len(gts)}"

	gt_selected = [False for _ in range(len(gts) // 2)]
	neg_preds = []

	for i in range(0, len(preds), 2):
		flag = False 				# False indicates current node doesn't fall in any circle
		for j in range(0, len(gts), 2):
			if (abs(int(preds[i]) - int(gts[j])) + abs(int(preds[i+1]) - int(gts[j+1]))) <= r:
				flag = True
				gt_selected[j // 2] = True
		if not flag:
			neg_preds.append([int(preds[i]), int(preds[i+1])])

	TP = sum(gt_selected)
	FN = len(gts) // 2 - TP
	FP = 0
	
	if len(neg_preds) > 0:
		neg_preds = np.array(neg_preds)		# neg_preds size should be (n, 2)
		clustering = DBSCAN(eps=32, metric='manhattan', min_samples=1).fit(neg_preds)
		FP = len(np.unique(clustering.labels_))

	return TP, FN, FP


def cal_F1(preds, gts, image_name):
	# preds and gts format example = [['12', '203', '294', '1024'], ['39', '95', '283', '94']]

	statistics = {'TP':0, 'FP':0, 'FN':0}
	for i in range(len(preds)):
		TP, FN, FP = cal_TPFNFP(preds[i], gts[i])		# for each image of a patient
		statistics['TP'] += TP
		statistics['FN'] += FN
		statistics['FP'] += FP
	F1_score = float(2 * statistics['TP']) / (2 * statistics['TP'] + statistics['FN'] + statistics['FP'])

	return statistics['TP'], statistics['FN'], statistics['FP'], F1_score

import torchvision


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer,num_classes)
    return model

def get_pos_patient(config:Config):
  test_dict = {"idx":[], "path":[], "patient":[]}
  test_dir = config.test_dir
  for pat_dir in os.listdir(test_dir):
    for ct_img in os.listdir(os.path.join(test_dir, pat_dir)):
      test_dict["idx"].append(ct_img[:ct_img.find('.')])
      test_dict["path"].append(os.path.join(pat_dir, ct_img))
      test_dict["patient"].append(pat_dir)

  test_df = pd.DataFrame(test_dict)
  test_df["labels"] = [2]*len(test_df)
  csv_df = pd.read_csv(config.case_level_csv)
  test_df = test_df.set_index("idx")
  test_df.loc[csv_df["id"], "labels"] = csv_df['label'].to_list()
  test_df = test_df.reset_index()
  return test_df.loc[test_df["labels"]!=0].reset_index(), csv_df



  