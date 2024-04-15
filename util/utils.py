import numpy as np
import random
import torch
import os
import cv2
from sklearn.utils.class_weight import compute_class_weight
import pickle
from pathlib import Path

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def calc_crack_pixel_weight(data_dir):
    print('Computing class weights...')

    cweight_path = data_dir + '/cweight.pkl'

    if os.path.exists(cweight_path):
        print('Loading saved class weights.')
        with open(cweight_path, 'rb') as f:
            weight = pickle.load(f)
    else:
        files = []

        for path in Path(data_dir + '/SegmentationClass').glob('*.*'):
            label = cv2.imread(str(path)).astype(np.uint8)
            if 2 not in np.unique(label):
                files.append(label)
            
        all_arr = np.stack(files, axis=0)[:,:,:,0]

        weight = compute_class_weight(class_weight = 'balanced',classes=np.unique(label), y=all_arr.flatten())

        with open(cweight_path, 'wb') as f:
            pickle.dump(weight, f)
        print('Saved class weights under dataset path.')

    return weight

class mIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return np.nanmean(iu)
