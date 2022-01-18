import pandas as pd

from typing import Dict, List

class Slice:
  def __init__(self, index, series, label, gt_label, slice_id, coord, gt_coord) -> None:
    self.index = index
    self.series = series
    self.label = label
    self.gt_label = gt_label
    self.slice_id = slice_id
    self.coord = coord
    self.gt_coord = gt_coord
  def __eq__(self, other) -> bool:
    return self.slice_id == other.slice_id
  
  def __ge__(self, other) -> bool:
    return self.slice_id>=other.slice_id
  

class Result:
  def __init__(self, data_df:pd.DataFrame) -> None:
    self.data_df = data_df
    self.result:Dict[int, List[Slice]] = dict()
  
  def add(self, ids, predicts, gt_labels, coords, gt_coords):
    for i in range(len(ids)):
      index = ids[i]
      s = Slice(i, self.data_df.loc[index, 'series'],predicts[i],gt_labels[i], self.data_df.loc[index, 'idx'], coords[i], gt_coords[i])
      self.result.setdefault(self.data_df.loc[index, 'series'], []).append(s)
    
  
  def assign_label(self):
    for key, item in self.result.items():
      fraction = False
      for slice in item:
        if slice.label == 1:
          fraction = True
      if fraction:
        print("find fraction")
        for slice in item:
          if slice.label==0:
            slice.label = -1
      item.sort(key=lambda x:x.slice_id)
  
  def coord_to_str(self, coords):
    s = ''
    for x, y in coords:
      s += ' ' if len(s) > 0 else ''
      s += str(x)
      s += ' '
      s += str(y)
    return s
  
  def to_csv(self, csv_path):
    with open(csv_path, 'w') as fout:
      fout.write("id,label,coords\n")
      for value in self.result.values():
        for slice in value:
          fout.write(','.join([slice.slice_id, str(slice.label), self.coord_to_str(slice.coord)]))
          fout.write('\n')

  def cal_acc(self):
    corrects = 0
    cnt = 0
    for value in self.result.values():
      for slice in value:
        cnt += 1
        if slice.label == slice.gt_label:
          corrects += 1
    print("Accuracy: [{}/{}] {:.2f}%".format(corrects, cnt, 100*corrects/cnt))
    return corrects/cnt
