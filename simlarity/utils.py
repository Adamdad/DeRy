import json
import mmcv
import numpy as np
import os
import pickle
import timm
import torch
from compare_functions import (cca_torch, cka_linear_torch, cka_rbf_torch, lr_torch)
from feature_extraction import (create_sub_network,
                                create_sub_network_transformer,
                                get_graph_node_names, graph_to_table)
from torchvision import models

import third_package.timm as mytimm
from blocklize import MODEL_BLOCKS, MODEL_STATS, MODEL_ZOO, MODEL_PRINT
from blocklize.block_meta import MODEL_INOUT_SHAPE
import mmcls


def create_feature_dict(path):
    result_dict = dict()
    for name in os.listdir(path):
        tmp_dict = mmcv.load(os.path.join(path, name))
        for ink in tmp_dict.keys():
            if ink in result_dict.keys():
                for outk in tmp_dict[ink].keys():
                    result_dict[ink][outk] = torch.cat(
                        [tmp_dict[ink][outk], result_dict[ink][outk]], dim=0)
            else:
                result_dict[ink] = tmp_dict[ink]
    return result_dict


def similarity_pair(pickle1, pickle2):
    feat1 = mmcv.load(pickle1)
    feat2 = mmcv.load(pickle2)
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())
    print(f'number of layers in {pickle1} is {num_layer1}')
    print(f'number of layers in {pickle2} is {num_layer2}')

    cka_map = torch.zeros((num_layer1, num_layer2))
    prog_bar = mmcv.ProgressBar(num_layer1)
    for i, (k1, v1) in enumerate(feat1.items()):
        for j, (k2, v2) in enumerate(feat2.items()):
            cka_from_examples = cka_linear_torch(v1, v2)
            cka_map[i, j] = cka_from_examples
            print(f'layers {i} in {pickle1} and layers {j} in {pickle2}')
        prog_bar.update()
    print(cka_map)


def network_to_module(model_name, block_name, backend):
    if backend == 'timm':
        backbone = timm.create_model(
            model_name, pretrained=True, scriptable=True)
    elif backend == 'pytorch':
        backbone = getattr(models, model_name)(pretrained=True)

    for name, module in backbone.named_modules():
        if name == block_name:
            return module


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # if isinstance(obj, int64):
        return super(NpEncoder, self).default(obj)


def network_to_module_subnet(model_name, block_input, block_output, backend):
    if backend == 'timm':
        backbone = timm.create_model(
            model_name, pretrained=True)
    elif backend == 'mytimm':
        backbone = mytimm.create_model(
            model_name, pretrained=True)
    elif backend == 'mmcv':
        config = MODEL_STATS[model_name]['cfg']
        cfg = mmcv.Config.fromfile(config)
        backbone = mmcls.build_backbone(cfg.model.backbone)
    elif backend == 'pytorch':
        backbone = getattr(models, model_name)(pretrained=True)

    if isinstance(block_input, str):
        block_input = [block_input]
    elif isinstance(block_input, tuple):
        block_input = list(block_input)
    elif isinstance(block_input, list):
        block_input = block_input
    else:
        TypeError('Block input should be a string or tuple or list')

    if isinstance(block_output, str):
        block_output = [block_output]
    elif isinstance(block_output, tuple):
        block_output = list(block_output)
    elif isinstance(block_output, list):
        block_output = block_output
    else:
        TypeError('Block output should be a string or tuple or list')

    if model_name.startswith('swin_') or model_name.startswith('vit'):
        subnet = create_sub_network_transformer(backbone, model_name, block_input, block_output)
    else:
        subnet = create_sub_network(backbone, block_input, block_output)
    return subnet


class Block:
    def __init__(self, model_name, block_index, node_list):
        assert isinstance(model_name, str)
        assert isinstance(node_list, list)
        for i in range(len(node_list)-1):
            assert node_list[i+1] - node_list[i] == 1, node_list
        self.model_name = model_name
        self.block_index = block_index
        self.node_list = node_list
        # print(model_name)
        self.value = 0  # MODEL_STATS[self.model_name]['top1']
        self.size = 0
        self.group_id = None

    def print_split(self):
        start = self.node_list[0]
        end = self.node_list[-1]
        return [MODEL_STATS[self.model_name]['arch'],
                MODEL_BLOCKS[self.model_name][start],
                MODEL_BLOCKS[self.model_name][end],
                MODEL_STATS[self.model_name]['backend']]

    def get_inout_size(self):
        start = self.node_list[0]
        end = self.node_list[-1]
        start_name = MODEL_BLOCKS[self.model_name][start]
        end_name = MODEL_BLOCKS[self.model_name][end]
        self.in_size = MODEL_INOUT_SHAPE[self.model_name]['in_size'][start_name]
        self.out_size = MODEL_INOUT_SHAPE[self.model_name]['out_size'][end_name]

    def __len__(self):
        return len(self.node_list)

    def get_model_size(self):
        model, block_input, block_output, backend = self.print_split()
        block = network_to_module_subnet(
            model, block_input, block_output, backend)
        self.size = sum(p.numel() for p in block.parameters())/1e6

    def __eq__(self, other):
        if isinstance(other, Block):
            return (self.model_name == other.model_name and
                    self.block_index == other.block_index and
                    self.node_list == other.node_list)
        else:
            return False

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, cls=NpEncoder, indent=2)

    def __str__(self):
        nodes_in = str(self.node_list[0]) 
        node_out = str(self.node_list[-1])
        return f'{MODEL_PRINT[self.model_name]}:{nodes_in}-{node_out} Stage-{self.block_index}'
        # return f'Model Name:{self.model_name}\tNode list:{self.node_list}\tBlock Index: {self.block_index}\t Size:{self.size}'


class Block_Sim:
    def __init__(self, sim_dict):
        self.sim_dict = sim_dict

    def get_sim(self, block1, block2):
        if isinstance(block1, Block) and isinstance(block2, Block):
            # print(block1, block2)
            key = f'{block1.model_name}.{block2.model_name}'
            if key in self.sim_dict.keys():
                if block1 == block2:
                    block_sim = 1
                elif block1.model_name == block1.model_name and block1.block_index != block2.block_index:
                    block_sim = 0
                else:
                    sim_map = self.sim_dict[key]
                    try:
                        block_sim = (sim_map[block1.node_list[0], block2.node_list[0]] +
                                     sim_map[block1.node_list[-1], block2.node_list[-1]])
                    except:
                        AssertionError('The functional similarity can not be computed')
            else:
                block_sim = 0

            return block_sim
        else:
            TypeError('block 1 and block 2 must be Block instance')


class Block_Assign:
    def __init__(self, assignment_index, block_split_dict, centers):
        self.block2center = dict()
        self.center2block = [[c]for c in centers]
        self.centers = centers

        for m, model_name in enumerate(MODEL_ZOO):
            self.block2center[model_name] = dict()
            for j, block in enumerate(block_split_dict[model_name]):
                center_index = assignment_index[m, j]
                block.group_id = center_index
                self.block2center[model_name][j] = centers[center_index]
                self.center2block[center_index].append(block)

    def get_center(self, block):
        return self.block2center[block.model_name][block.block_index]

    def print_center(self):
        return '.'.join([str(c) for c in self.centers])

    def print_assignment(self):
        results = ''
        for i, group in enumerate(self.center2block):
            results += 'Center {}\n'.format(str(self.centers[i]))
            results += '\n'.join(['\t'+str(c) for c in group])
            results += '\n'
        print(results)

    def get_size(self):
        for group in self.center2block:
            for block in group:
                # block.get_model_size()
                block.get_inout_size()
                print(block)

    def save_assignment(self, out_file):
        with open(out_file, 'wb') as file:
            pickle.dump(self, file)


if __name__ == '__main__':
    # similarity_pair_blocks('/home/yangxingyi/InfoDrop/simlarity/out/resnet18/',
    #                        '/home/yangxingyi/InfoDrop/simlarity/out/resnet18_rand/')
    with open(f'test.pickle', 'rb') as file2:
        s1 = pickle.load(file2)
    s1.save_assignment()
