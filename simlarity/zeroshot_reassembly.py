import argparse
import copy
from tkinter.messagebox import NO
import mmcv
import pickle
import random
import torch
from mmcv import Config
from utils import MODEL_ZOO, Block, Block_Assign, Block_Sim

from mmcls.datasets.builder import build_dataloader, build_dataset
from simlarity.model_creater import Model_Creator
from simlarity.zero_nas import ZeroNas
from mmcv.cnn.utils import get_model_complexity_info


input_shape = (3, 224, 224)


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument(
        '--path', default='/home/yangxingyi/InfoDrop/simlarity/out/assignment/assignment_cnn_4.pkl')
    parser.add_argument('--C', type=float, default=30.)
    parser.add_argument('--minC', type=float, default=0.)
    parser.add_argument('--flop_C', type=float, default=10.)
    parser.add_argument('--minflop_C', type=float, default=0.)
    parser.add_argument('--trial', type=int, default=10)
    parser.add_argument('--num_batch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--zero_proxy', type=str,
                        choices=['jacov', 'grad_norm', 'naswot', 'synflow', 'snip', 'fisher'])
    parser.add_argument('--data_config', type=str,
                        default='configs/_base_/datasets/imagenet_bs64.py')

    args = parser.parse_args()

    args.maxC = args.C
    args.maxflop_C = args.flop_C

    return args


def check_valid(selected_block):
    cnn_max = 0
    vit_min = len(selected_block)
    for s in selected_block:
        if s is not None:
            if (s.model_name.startswith('vit') or s.model_name.startswith('swin')):
                if s.block_index < vit_min:
                    vit_min = s.block_index
            else:
                if s.block_index > cnn_max:
                    cnn_max = s.block_index

    return cnn_max < vit_min


def main():
    args = parse_args()
    with open(args.path, 'rb') as file:
        assignment = pickle.load(file)
    assert isinstance(assignment, Block_Assign)
    all_blocks = []
    for group in assignment.center2block:
        all_blocks.extend(group)
    all_blocks = [b for b in all_blocks if b.model_name in MODEL_ZOO]

    distributed = False
    data_cfg = Config.fromfile(args.data_config)
    dataset = build_dataset(data_cfg.data.train)
    data_cfg.data.samples_per_gpu = args.batch_size
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=data_cfg.data.samples_per_gpu,
        workers_per_gpu=data_cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)
    print('*'*10 + 'Dataloader Created' + '*'*10)
    indicator = ZeroNas(dataloader=data_loader,
                        indicator=args.zero_proxy,
                        num_batch=args.num_batch)
    creator = Model_Creator()

    best_value = 0
    best_size = 0
    best_selected_block = None
    K = len(assignment.center2block)
    for k in range(args.trial):
        # all_blocks = list(sorted(all_blocks, key=lambda x: x.value / x.size, reverse=True))
        random.shuffle(all_blocks)
        selected_group = [0 for _ in range(K)]
        selected_block_index = [0 for _ in range(K)]
        select_blocks = [None for _ in range(K)]
        iter_best_value = 0
        iter_best_size = 0
        for block in all_blocks:

            if selected_group[block.group_id] == 0 and selected_block_index[block.block_index] == 0:
                # No block has been selected at this position
                new_select = copy.deepcopy(select_blocks)
                new_select[block.block_index] = block
                if check_valid(new_select):
                    select_blocks = new_select
                    selected_group[block.group_id] = 1
                    selected_block_index[block.block_index] = 1

            else:
                new_select = copy.deepcopy(select_blocks)
                # check repeat and remove
                for i, b in enumerate(select_blocks):
                    if b is not None:
                        if b.block_index == block.block_index or b.group_id == block.group_id:
                            new_select[block.block_index] = None
                            selected_block_index[b.block_index] = 0
                            selected_group[b.group_id] = 0

                # append new block in
                new_select[block.block_index] = block
                selected_block_index[block.block_index] = 1
                selected_group[block.group_id] = 1
                # if check_valid(new_select):
                model = creator.create_hybrid(new_select)
                if model is None:
                    continue

                try:
                    new_flops, new_size = get_model_complexity_info(
                        model, input_shape, print_per_layer_stat=False, as_strings=False)
                    new_flops = round(new_flops / 10.**9, 3)
                    new_size = sum(p.numel() for p in model.parameters())/1e6
                except:
                    continue

                if new_size <= args.maxC and new_size > args.minC and new_flops <= args.maxflop_C and new_flops > args.minflop_C:
                    pass
                else:
                    print(
                        f'current size {new_size}M, current flops {new_flops}G, \tParam Range ({args.minC}M,{args.maxC}M), \tFLOPs Range ({args.minflop_C}GFLOPs,{args.maxflop_C}GFLOPs)')
                    continue

                # try:
                new_value = indicator.get_score(model)[args.zero_proxy]
                # except:
                #     continue

                print(
                    f'Current score {new_value}, current size {new_size}M, current flops {new_flops}G')
                del model
                torch.cuda.empty_cache()
                if new_value > iter_best_value and check_valid(new_select):
                    iter_best_value = new_value
                    iter_best_size = new_size
                    select_blocks = new_select

        if iter_best_value > best_value:
            best_value = iter_best_value
            best_size = iter_best_size
            best_selected_block = select_blocks
            print(
                f"[Iteration {k}], New best_value {best_value}, New size {best_size}, capacity {args.C}")
        else:
            print(
                f"[Iteration {k}], best_value {best_value}, size {best_size}, current value {iter_best_value}, No update")

    print(best_selected_block)
    best_selected_block = list(
        sorted(best_selected_block, key=lambda x: x.block_index))
    model = creator.create_hybrid(best_selected_block)
    assert model is not None, "Searched model can not be none"
    size = sum(p.numel() for p in model.parameters())/1e6
    print(f'Final size {size}, capacity {args.C}')
    best_model_cfg = creator.create_hybrid_cfg(best_selected_block)

    dataname = data_cfg.data.train.type
    file_name = f'resaemly_C-{args.C}_FLOPsC-{args.flop_C}_zeroproxy_hybrid_{dataname}-{args.zero_proxy}.py'
    best_model_cfg = Config(dict(model=best_model_cfg))
    print(best_model_cfg.pretty_text)

    with open(file_name, 'w') as f:
        f.write(best_model_cfg.pretty_text)


if __name__ == '__main__':
    main()
