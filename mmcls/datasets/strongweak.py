
import copy
import json
from mmcv.runner.dist_utils import get_dist_info
import numpy as np
import os.path as osp
import torch
from functools import total_ordering
from torch.utils.data import Dataset
from tqdm import tqdm

from .builder import DATASETS, build_dataset
from .pipelines import Compose


@DATASETS.register_module()
class StrongWeakDataset(Dataset):

    def __init__(self,
                 data_cfg,
                 strong_pipeline=None):
        super(StrongWeakDataset, self).__init__()

        self.dataset = build_dataset(data_cfg)
        self.CLASSES = self.dataset.CLASSES
        if isinstance(strong_pipeline, list):
            self.strong_pipeline = Compose(strong_pipeline)
        else:
            self.strong_pipeline = None

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset.data_infos[idx])
        copy_results = copy.deepcopy(self.dataset.data_infos[idx])
        results = self.dataset.pipeline(results)
        strong_results = self.strong_pipeline(copy_results)

        return dict(img=results['img'],
                    strong_img=strong_results['img'],
                    gt_label=results['gt_label'],
                    rand_gt_label=strong_results['gt_label'])
    def __len__(self):
        return len(self.dataset)
