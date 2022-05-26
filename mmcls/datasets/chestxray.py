import json
import numpy as np
import os
import torch
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
from torchvision.datasets import folder as dataset_parser

from mmcls.datasets.base_dataset import BaseDataset

from .builder import DATASETS, build_dataset
from mmcv.runner.dist_utils import get_dist_info

def make_dataset(dataset_root, split_file_path, task='All', pl_list=None):

    with open(split_file_path, 'r') as f:
        img = f.readlines()

    if task == 'semi_fungi':
        img = [x.strip('\n').rsplit('.JPG ') for x in img]
    # elif task[:9] == 'semi_aves':
    else:
        img = [x.strip('\n').rsplit() for x in img]

    # Use PL + l_train
    if pl_list is not None:
        if task == 'semi_fungi':
            pl_list = [x.strip('\n').rsplit('.JPG ') for x in pl_list]
        # elif task[:9] == 'semi_aves':
        else:
            pl_list = [x.strip('\n').rsplit() for x in pl_list]
        img += pl_list

    for idx, x in enumerate(img):
        if task == 'semi_fungi':
            img[idx][0] = os.path.join(dataset_root, x[0] + '.JPG')
        else:
            img[idx][0] = os.path.join(dataset_root, x[0])
        img[idx][1] = int(x[1])

    classes = [x[1] for x in img]
    
    num_classes = len(set(classes))
    rank, _ = get_dist_info()
    if rank == 0:
        print('# images in {}: {}'.format(split_file_path, len(img)))
    return img, num_classes


@DATASETS.register_module()
class ChestXray14Dataset(BaseDataset):
    def __init__(self,
                 data_prefix,
                 pipeline,
                 task,
                 pl_list=None,
                 classes=None,
                 ann_file=None,
                 test_mode=False):

        self.task = task
        self.pl_list = pl_list
        super().__init__(data_prefix,
                         pipeline,
                         classes=classes,
                         ann_file=ann_file,
                         test_mode=test_mode)
        

    def load_annotations(self):
        self.samples, self.num_classes = make_dataset(self.data_prefix,
                                                      self.ann_file, self.task, pl_list=self.pl_list)

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': None}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos