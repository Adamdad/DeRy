
import argparse
import mmcv
import timm
from feature_extraction import create_sub_network, get_graph_node_names, create_sub_network_transformer, graph_to_table
from torchvision import models
import torch
from mmcls.models.builder import build_classifier

import third_package.timm as mytimm
from utils import network_to_module_subnet
from utils import network_to_module
from mmcls.models import build_backbone

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model_name', default='resnext50_32x4d')
    parser.add_argument('--backend', default='timm',
                        choices=['timm', 'pytorch', 'mytimm', 'mmcv'])
    parser.add_argument('--print_only', action='store_true')
    parser.add_argument('--module_names', type=list)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--config',  default='')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    model_name = args.model_name

    if args.backend == 'timm':
        backbone = timm.create_model(
            model_name, pretrained=True, scriptable=True)
    elif args.backend == 'pytorch':
        backbone = getattr(models, model_name)(pretrained=True)

    feature_extractor = create_sub_network(
	    backbone, input_nodes=['layer1.0'], return_nodes=['layer2.1'])
    print(feature_extractor)

def show_node():
    args = parse_args()
    model_name = args.model_name

    if args.backend == 'timm':
        backbone = timm.create_model(
            model_name, pretrained=False)
    elif args.backend == 'mytimm':
         backbone = mytimm.create_model(
            model_name, pretrained=False)
    elif args.backend == 'pytorch':
        backbone = getattr(models, model_name)(pretrained=False)
    elif args.backend == 'mmcv':
        cfg = mmcv.Config.fromfile(args.config)
        backbone = build_backbone(cfg.model.backbone)


    for key, m in backbone.named_modules():
        print(key)
    input = torch.zeros((1, 3, 160, 160))
    output = backbone(input)
    # gm = torch.fx.symbolic_trace(backbone)

    
    # tab = graph_to_table(gm.graph)
    # with open(f'{model_name}_table.txt', 'w')  as f:
    #     f.write(tab)
    # train_nodes, eval_nodes = get_graph_node_names(backbone)
    # # print(train_nodes)
    # # ## 
    # def sort_key(string):
    #     segments = string.split('.')
    #     # print(segments)
    #     return int(segments[1]) * 100 + int(segments[3])
    # node_names = []
    # for node_name in train_nodes:
    #     if node_name.startswith('trunk_output.block'):
    #         nodes = node_name.split('.')[:3]
    #         if len(nodes) == 3:
    #             node_name = '.'.join(nodes)
    #             node_names.append(node_name)
    # node_names = list(set(node_names))
    # node_names.sort()
    # print(node_names)

def main_full_cfg():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    model = build_classifier(cfg.model)
    input = torch.zeros((1, 3, 320, 320))
    output = model.extract_feat(input)

def main_cfg():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    block_list = cfg.model.backbone.block_list
    output = ''
    for node_info in block_list:
        if len(node_info) == 4:
            model, block_input, block_output, backend = node_info
        elif len(node_info) == 5:
            model, block_input, block_output, backend, _ = node_info
        block = network_to_module_subnet(model, block_input, block_output, backend)
        print(node_info)
        output += str(block) 
        output += '\n'
    with open(f'subnet.log', 'w')  as f:
        f.write(output)

def main_cfg_inout():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    block_list = cfg.model.backbone.block_list
    channels = [cfg.model.backbone.base_channels]
    channels += [a['output_channel']for a in cfg.model.backbone.adapter_list]
    output = ''
    for node_info, in_cha in zip(block_list, channels):
        if len(node_info) == 4:
            model, block_input, block_output, backend = node_info
        elif len(node_info) == 5:
            model, block_input, block_output, backend, _ = node_info
        input = torch.zeros((1, in_cha, 32, 32))
        block = network_to_module_subnet(model, block_input, block_output, backend)
        netoutput = block(input)
        if isinstance(netoutput, dict):
            netoutput = list(netoutput.values())[0]
        print('Input', input.shape, 'Output', netoutput.shape)
        print(node_info)
        output += str(block) 
        output += '\n'
    with open(f'subnet.log', 'w')  as f:
        f.write(output)


if __name__ == '__main__':
    # show_node()
    # main()

    # main_cfg()
    main_full_cfg()
    # main_cfg_inout()
    