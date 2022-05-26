import re
from tokenize import group
import numpy as np
import argparse
from utils import Block, Block_Assign, Block_Sim, MODEL_ZOO
import pickle
import copy
import random
# groups is list of integers in ascending order without gaps


def multipleChoiceKnapsack(W, weights, values, groups):
    n = len(values)
    # Use a matrix to record the weight from 0 to W and item from 0 to n
    K = np.zeros((n+1, W+1))
    selected = []
    for w in range(W+1):
        for i in range(n+1):
            if i == 0 or w == 0:
                K[i,w] = 0
            elif weights[i-1] <= w:
                sub_max = 0

                prev_group = groups[i-1]-1
                sub_K = K[:, w-weights[i-1]]
                for j in range(n+1):
                    if groups[j-1] == prev_group and sub_K[j] > sub_max:
                        sub_max = sub_K[j]
                # find the previous best sub results
                K[i,w] = max(sub_max+values[i-1], K[i-1,w])
            else:
                K[i,w] = K[i-1,w]
    print(K)
    return K[n][W]


# Example
# values = [60, 100, 120, 110]
# weights = [10, 20, 30, 20]
# groups = [0, 1, 2, 1]
# W = 20
# print(multipleChoiceKnapsack(W, weights, values, groups))  # 220

def greedyMCKP(W, blocks):
    """AI is creating summary for greedyMCKP

    Args:
        W ([int]): the maximum capacity
        blocks ([List of Block]): a list of nodes that can be picked
    """
def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('--path', default='/home/yangxingyi/InfoDrop/simlarity/out/assignment/assignment_4.pkl')
    parser.add_argument('--C', type=float, default=10.)
    parser.add_argument('--trial', type=int, default=10)

    args = parser.parse_args()

    return args

def is_cnn_after(block, selected_block):
    for s in selected_block:
        if s is not None:
            if s.model_name.startswith('vit') and  s.model_name.startswith('swin'):
                pass
            else:
                
                return True
    return False

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
                
    # print(cnn_max, vit_min,selected_block)
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
        
    best_value = 0
    best_selected_block = None
    K = len(assignment.center2block)
    for k in range(args.trial):
        # all_blocks = list(sorted(all_blocks, key=lambda x: x.value / x.size, reverse=True))
        random.shuffle(all_blocks)
        selected_group = [0 for _ in range(K)]
        selected_block_index = [0 for _ in range(K)]
        select_blocks = [None for _ in range(K)]
        iter_best_value = 0
        for block in all_blocks:

            if selected_group[block.group_id]==0 and selected_block_index[block.block_index]==0:
                new_select = copy.deepcopy(select_blocks)

                new_select[block.group_id] = block
                new_size = sum([b.size for b in new_select if b is not None])
                if new_size < args.C and check_valid(new_select):
                    select_blocks = new_select
                    selected_group[block.group_id] = 1
                    selected_block_index[block.block_index] = 1
            

            else:
                new_select = copy.deepcopy(select_blocks)
                # check repeat and remove
                for i, b in enumerate(select_blocks):
                    if b is not None:
                        if b.block_index == block.block_index or b.group_id == block.group_id:
                            new_select[i] = None
                            selected_block_index[b.block_index] = 0
                            selected_group[b.group_id] = 0

                # append new block in
                new_select[block.group_id] = block
                selected_block_index[block.block_index] = 1
                selected_group[block.group_id] = 1
                new_size = sum([b.size for b in new_select if b is not None])
                new_value = sum([b.value for b in new_select if b is not None])
                if new_value > iter_best_value and new_size < args.C and check_valid(new_select):
                    iter_best_value = new_value
                    select_blocks = new_select

        if iter_best_value > best_value:
            best_value = iter_best_value
            best_selected_block = select_blocks
            size = sum([b.size for b in best_selected_block if b is not None])
            print(f"[Iteration {k}], New best_value {best_value}, size {size}, capacity {args.C}")
        else:
            print(f"[Iteration {k}], best_value {best_value}, current value {iter_best_value}, No update")
            
    print(best_selected_block)
    print(check_valid(best_selected_block))
    size = sum([b.size for b in best_selected_block if b is not None])
    print(f'Final size {size}, capacity {args.C}')
    best_selected_block = list(sorted(best_selected_block, key=lambda x: x.block_index))
    for b in best_selected_block:
        split = b.print_split()
        print(split)

            


if __name__ == '__main__':
    main()
    