import torch
import torch.nn as nn
from torch.optim import optimizer
from torch.utils.data import DataLoader
import os

import torchvision
from config import Config
from validation.val import predict_coord
from utils import gen_target

from typing import Dict, Union

class Worker:
  def __init__(self, config:Config, model,train_df, val_df, dataloaders:Dict[str, DataLoader], device=torch.device("cuda")) -> None:
    self.model:Union[nn.Module, torchvision.models.detection.FasterRCNN] = model
    self.config = config
    self.train_dataloader = dataloaders['train']
    self.val_dataloader = dataloaders['val']
    self.device = device
    self.optimizer = torch.optim.SGD(self.model.parameters(), config.train_lr, 0.9, weight_decay=1e-5)
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config.step_size, config.gamma)
    self.train_loss_his = {"loss_classifier":[], "loss_box_reg": [], "loss_objectness": [], "loss_rpn_box_reg":[], "total":[]}
    self.train_cls_acc_his = list()
    self.val_cls_acc_his = list()
    self.train_df = train_df
    self.val_df = val_df
    self.thres = config.box_score_threshold

  def box_predict(self):
    self.model = self.model.to(self.device)

    for epoch in range(self.config.n_epoch):
      self.model.train()
      running_loss = {"loss_classifier":0.0, "loss_box_reg": 0.0, "loss_objectness":0.0, "loss_rpn_box_reg":0.0}
      print(f"Epoch {epoch+1}")
      print("============================")
      for idx, (imgs, _, ind) in enumerate(self.train_dataloader):
        coords = self.train_df.loc[ind, "coords"].tolist()
        targets = gen_target(coords)
        imgs = imgs.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(imgs, targets)

        loss = output["loss_classifier"] + 10 * output["loss_box_reg"] + output["loss_objectness"] + 10 * output["loss_rpn_box_reg"]
        
        loss.backward()
        self.optimizer.step()

        for key in running_loss:
          running_loss[key] += output[key].item()

        if (idx+1) % (len(self.train_dataloader) // self.config.log_interval) == 0:
          print(f"[{idx+1}/{len(self.train_dataloader)}] ({100*(idx+1)/len(self.train_dataloader):.2f}%) Train Loss: {loss:.4f}")
      if self.config.use_scheduler:
        self.scheduler.step()
      total = 0
      for key in running_loss:
        total += running_loss[key]
        self.train_loss_his[key].append(running_loss[key]/(len(self.train_dataloader)))
      self.train_loss_his["total"].append(total/len(self.train_dataloader))

      f1_score = predict_coord(self.config, self.model, self.val_df, self.val_dataloader, self.device)
      self.val_cls_acc_his.append(f1_score)

      if (epoch+1) % self.config.save_interval == 0:
        torch.save(self.model.state_dict(), os.path.join(self.config.ckpt_dir, f"epoch_{epoch+1}.pth"))
  