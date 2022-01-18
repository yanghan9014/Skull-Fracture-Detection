import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.vgg import vgg16
from config import Config

class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
    super(ConvBlock, self).__init__()
    padding = int((kernel_size - 1) / 2) if same_padding else 0
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
    self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
    self.relu = nn.ReLU(inplace=True) if relu else None

  def forward(self, x):
    x = self.conv(x)
    if self.bn is not None:
      x = self.bn(x)
    if self.relu is not None:
      x = self.relu(x)
    return x

class SkullRCNN(nn.Module):
  _feat_stride = [16, ]
  def __init__(self, config:Config, backbone, training:bool):
    super(SkullRCNN, self).__init__()
    self.feature = backbone
    self.conv = ConvBlock(512, 512, 3, same_padding=True)
    self.cls_cnt = len(config.anchor_scale)*len(config.bbox_lw_ratio)*2
    self.bbox_cnt = len(config.anchor_scale)*len(config.bbox_lw_ratio)*2
    self.cls_conv = ConvBlock(512, self.cls_cnt, kernel_size=1, relu=False)
    self.bbox_conv = ConvBlock(512, self.bbox_cnt, kernel_size=1, relu=False)
    self.training = training
  @staticmethod
  def reshape(x, d):
    input_shape = x.size()
    x = x.view(
      input_shape[0],
      int(d),
      int(float(input_shape[1] * input_shape[2]) / float(d)),
      input_shape[3]
    )
    return x

  def forware(self, x):
    feature = self.feature(x)                               # [bs, 512, 7, 7]
    rpn_conv = self.conv(feature)                           # [bs, 512, 7, 7]
    rpn_cls_score = self.cls_conv(rpn_conv)                 # [bs, 9*2, 7, 7]
    rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)  # [bs, 2, 9*7, 7]
    rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
    rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.cls_cnt)

    rpn_bbox_pred = self.bbox_conv(rpn_conv)
    return rpn_cls_prob, rpn_bbox_pred


class RCNNLoss(nn.Module):
  def __init__(self, Lambda:float):
    super(RCNNLoss, self).__init__()
    self.Lambda = Lambda
  
  def forward(self, cls_pred, coord_pred, cls_label, coord_label):
    cls_loss = nn.BCELoss(cls_pred, cls_label)
    coord_loss = torch.pow(coord_pred - coord_label).sum(1) / coord_pred.size(0)
    return cls_loss + self.Lambda * coord_loss

    
    
  