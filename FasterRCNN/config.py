import os

class Config:
  def __init__(self) -> None:
    self.workspace = f'.'
    self.train_dir = os.path.join('skull', 'train')
    self.test_dir = os.path.join('skull', 'test')
    self.train_json = os.path.join('skull', 'records_train.json')
    self.case_level_csv = "../out.csv"
    self.output_csv_path = os.path.join(self.workspace, 'out.csv')
    self.ckpt_dir = os.path.join(self.workspace, 'checkpoints')
    self.anchor_sizes = ((8, 16, 32, 64),)
    self.anchor_aspect_ratios = ((0.5, 1.0, 2.0),)
    self.roi_output_size = 7
    self.box_score_threshold = 0.1
    self.image_size = (512, 512)

    self.n_epoch = 20
    self.batch_size = 1
    self.train_lr = 1e-3
    self.betas = (0.5, 0.999)
    self.use_scheduler = True
    self.step_size = 10
    self.gamma = 0.5

    self.log_interval = 10
    self.save_interval = 5
    
  
  def init(self):
    os.makedirs(self.workspace, exist_ok=True)
    os.makedirs(self.ckpt_dir, exist_ok=True)
  
  def printInfo(self):
    print(f"Workspace: {self.workspace}")
    print(f"anchor size: {self.anchor_sizes}")
    print(f"anchor aspect ratio: {self.anchor_aspect_ratios}")
    print(f"score threshold: {self.box_score_threshold}")
    print(f"n_epoch: {self.n_epoch}")
    print(f"Batch size: {self.batch_size}")
    print(f"learning rate: {self.train_lr}")
    print(f"use scheduler: {self.use_scheduler}")
    
    