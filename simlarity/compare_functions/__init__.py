from .cca_torch import cca_torch
from .cka_torch import cka_linear_torch, cka_rbf_torch
from .lr_torch import lr_torch
import mmcv
import torch
import numpy as np


def similarity_pair_batch_cka(data1, data2, bs=2048):
    feat1 = data1['feat']
    feat2 = data2['feat']
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())
    name1 = data1['model_name']
    name2 = data2['model_name']
    print(f'number of layers in {name1} is {num_layer1}')
    print(f'number of layers in {name2} is {num_layer2}')
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())

    num_sample = list(feat1.values())[0].shape[0]
    num_batch = np.int64(np.ceil(num_sample / bs))
    print(
        f'number of samples {num_sample}, number of batch {num_batch}, batch size {bs}')

    cka_map = torch.zeros((num_batch, num_layer1, num_layer2)).cuda()
    prog_bar = mmcv.ProgressBar(num_batch)
    for b_id in range(num_batch):
        start = b_id*bs
        end = (b_id+1)*bs if (b_id+1)*bs < num_sample-1 else num_sample-1
        for i, (k1, v1) in enumerate(feat1.items()):
            for j, (k2, v2) in enumerate(feat2.items()):
                cka_from_examples = cka_linear_torch(
                    torch.tensor(v1[start:end]).cuda(),
                    torch.tensor(v2[start:end]).cuda())
                cka_map[b_id, i, j] = cka_from_examples
        prog_bar.update()
    return cka_map.mean(0).detach().cpu().numpy()

def similarity_pair_batch_rbf_cka(data1, data2, bs=2048):
    feat1 = data1['feat']
    feat2 = data2['feat']
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())
    name1 = data1['model_name']
    name2 = data2['model_name']
    print(f'number of layers in {name1} is {num_layer1}')
    print(f'number of layers in {name2} is {num_layer2}')
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())

    num_sample = list(feat1.values())[0].shape[0]
    num_batch = np.int64(np.ceil(num_sample / bs))
    print(
        f'number of samples {num_sample}, number of batch {num_batch}, batch size {bs}')

    cka_map = torch.zeros((num_batch, num_layer1, num_layer2)).cuda()
    prog_bar = mmcv.ProgressBar(num_batch)
    for b_id in range(num_batch):
        start = b_id*bs
        end = (b_id+1)*bs if (b_id+1)*bs < num_sample-1 else num_sample-1
        for i, (k1, v1) in enumerate(feat1.items()):
            for j, (k2, v2) in enumerate(feat2.items()):
                cka_from_examples = cka_rbf_torch(
                    torch.tensor(v1[start:end]).cuda(),
                    torch.tensor(v2[start:end]).cuda())
                cka_map[b_id, i, j] = cka_from_examples
        prog_bar.update()
    return cka_map.mean(0).detach().cpu().numpy()

def similarity_pair_batch_lr(data1, data2, bs=2048):
    feat1 = data1['feat']
    feat2 = data2['feat']
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())
    name1 = data1['model_name']
    name2 = data2['model_name']
    print(f'number of layers in {name1} is {num_layer1}')
    print(f'number of layers in {name2} is {num_layer2}')
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())

    num_sample = list(feat1.values())[0].shape[0]
    num_batch = np.int64(np.ceil(num_sample / bs))
    print(
        f'number of samples {num_sample}, number of batch {num_batch}, batch size {bs}')

    cka_map = torch.zeros((num_batch, num_layer1, num_layer2)).cuda()
    prog_bar = mmcv.ProgressBar(num_batch)
    for b_id in range(num_batch):
        start = b_id*bs
        end = (b_id+1)*bs if (b_id+1)*bs < num_sample-1 else num_sample-1
        for i, (k1, v1) in enumerate(feat1.items()):
            for j, (k2, v2) in enumerate(feat2.items()):
                cka_from_examples = lr_torch(
                    torch.tensor(v1[start:end]).cuda(),
                    torch.tensor(v2[start:end]).cuda())
                cka_map[b_id, i, j] = cka_from_examples
        prog_bar.update()
    return cka_map.mean(0).detach().cpu().numpy()

SIM_FUNC = {
    'cka':similarity_pair_batch_cka,
    'rbf_cka':similarity_pair_batch_rbf_cka,
    'lr':similarity_pair_batch_lr
}