import argparse
import copy
import mmcv
import numpy as np
import os
from itertools import combinations
from random import sample
from utils import Block, Block_Assign, Block_Sim

from blocklize import MODEL_BLOCKS, MODEL_ZOO


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument(
        '--sim_path', default='/home/yangxingyi/InfoDrop/simlarity/out/sim_cka/')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--trial', type=int, default=200)
    parser.add_argument(
        '--out', default='/home/yangxingyi/InfoDrop/simlarity/out/assignment')
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--num_iter', type=int, default=200)

    args = parser.parse_args()

    return args


def recenter(args, block_split_dict, block_sims, assignment):
    new_centers = []
    for c_id, group in enumerate(assignment.center2block):
        num_in_group = len(group)
        if num_in_group == 0:
            print(group)
            print(assignment.centers[c_id])
        group_sim = np.zeros((num_in_group, num_in_group))
        for b1_id in range(num_in_group):
            for b2_id in range(num_in_group):
                block1 = group[b1_id]
                block2 = group[b2_id]
                group_sim[b1_id, b2_id] = block_sims.get_sim(block1, block2)
        new_center_index = np.argmax(group_sim.sum(0))
        new_centers.append(group[new_center_index])

    assignment.centers = new_centers
    return assignment


def reassign(args, block_split_dict, block_sims, assignment):
    num_model = len(MODEL_ZOO)
    centers = assignment.centers
    # Assign each blocks to the K group with the maximum similarity
    block_sim_map = np.zeros((args.K, num_model, args.K))
    for i, center_block in enumerate(centers):
        for m, other_model_name in enumerate(MODEL_ZOO):
            for j, block in enumerate(block_split_dict[other_model_name]):

                block_sim = block_sims.get_sim(center_block, block)
                block_sim_map[i, m, j] = block_sim

    assignment_index = np.argmax(block_sim_map, axis=0)
    assignment = Block_Assign(assignment_index=assignment_index,
                              block_split_dict=block_split_dict,
                              centers=centers)
    return assignment


def compute_cost(block, all_assignmemt, block_sims):
    # find the block similarity with its corresponding center
    center_block = all_assignmemt.get_center(block)
    block_sim = block_sims.get_sim(center_block, block)
    return block_sim


def total_cost(assignment, block_sims):
    total_sim = 0
    for group in assignment.center2block:
        num_in_group = len(group)
        group_sim = np.zeros((num_in_group, num_in_group))
        for b1_id in range(num_in_group):
            for b2_id in range(b1_id+1, num_in_group):
                block1 = group[b1_id]
                block2 = group[b2_id]
                group_sim[b1_id, b2_id] = block_sims.get_sim(block1, block2)
        total_sim += np.sum(group_sim)
    return total_sim


def repartition(args, block_split_dict, block_sims, all_assignmemt):
    improved = False
    for m_id, model_name in enumerate(MODEL_ZOO):
        iter_block_split = copy.deepcopy(block_split_dict[model_name])
        for b_id in range(len(iter_block_split)-1):
            block1 = iter_block_split[b_id]
            block2 = iter_block_split[b_id+1]
            len1, len2 = len(block1), len(block2)
            # use the current assignment find the current score

            best_cost = (compute_cost(block1, all_assignmemt, block_sims) +
                         compute_cost(block2, all_assignmemt, block_sims))
            concat_nodes = (block1.node_list + block2.node_list)
            if len2 > block_split_dict['min_node'] and len2 <= block_split_dict['max_node']:
                block1 = Block(model_name, b_id, concat_nodes[:len1+1])
                block2 = Block(model_name, b_id+1, concat_nodes[len1+1:])
                new_cost = (compute_cost(block1, all_assignmemt, block_sims) +
                            compute_cost(block2, all_assignmemt, block_sims))
                if new_cost > best_cost:
                    improved = True
                    best_cost = new_cost
                    iter_block_split[b_id] = block1
                    iter_block_split[b_id+1] = block2

            if len1 > block_split_dict['min_node'] and len1 <= block_split_dict['max_node']:
                block1 = Block(model_name, b_id, concat_nodes[:len1-1])
                block2 = Block(model_name, b_id+1, concat_nodes[len1-1:])

                new_cost = (compute_cost(block1, all_assignmemt, block_sims) +
                            compute_cost(block2, all_assignmemt, block_sims))
                if new_cost > best_cost:
                    improved = True
                    best_cost = new_cost
                    iter_block_split[b_id] = block1
                    iter_block_split[b_id+1] = block2

        block_split_dict[model_name] = iter_block_split
    return block_split_dict, improved


def init_partition(args):
    all_node_list = dict()
    block_split_dict = dict()
    for model_name in MODEL_ZOO:
        node_list = MODEL_BLOCKS[model_name]
        all_node_list[model_name] = node_list
        N = len(node_list)
        node_per_block = int(np.floor(N/args.K))
        max_node_per_block = int(np.ceil(N/args.K) * (1+args.eps))
        min_node_per_block = 1  # max(N - (args.K-1) * max_node_per_block, 1)

        # print(
        #     f'[{model_name}]\tMax node: {max_node_per_block}, Min node {min_node_per_block}')

        block_split_dict['min_node'] = min_node_per_block
        block_split_dict['max_node'] = max_node_per_block

        block_split_dict[model_name] = []
        node_indexs = np.arange(N)
        node_split = list(sample(list(range(1, N-1)), args.K-1))
        node_split.sort()
        node_split = [0] + node_split
        for k in range(args.K):
            i1 = node_split[k]
            if k == args.K-1:
                block = Block(model_name, k, list(node_indexs[i1:]))
                block_split_dict[model_name].append(block)
            else:
                i2 = node_split[k+1]
                block = Block(model_name, k, list(
                    node_indexs[i1:i2]))
                block_split_dict[model_name].append(block)

        assert len(block_split_dict[model_name]) == args.K

    return block_split_dict


def print_partition(block_split_dict):
    for model_name in MODEL_ZOO:
        print(f"[{model_name}]")
        model_split = '.'.join([str(block)
                               for block in block_split_dict[model_name]])
        print(model_split)


def init_assign(args, block_split_dict, block_sims):
    # select K random blocks
    all_blocks = []
    num_model = len(MODEL_ZOO)
    for model_name in MODEL_ZOO:
        for i in range(args.K):
            block = block_split_dict[model_name][i]
            all_blocks.append(block)
    centers = sample(all_blocks, args.K)

    # Assign each blocks to the K group with the minimum
    block_sim_map = np.zeros((args.K, num_model, args.K))
    for i, center_block in enumerate(centers):
        for m, other_model_name in enumerate(MODEL_ZOO):
            for j, block in enumerate(block_split_dict[other_model_name]):
                block_sim = block_sims.get_sim(center_block, block)
                block_sim_map[i, m, j] = block_sim

    assignment_index = np.argmax(block_sim_map, axis=0)
    assignment = Block_Assign(assignment_index=assignment_index,
                              block_split_dict=block_split_dict,
                              centers=centers)

    return assignment


def get_all_sim(args):
    all_sim_dict = dict()
    comb = list(combinations(MODEL_ZOO, 2))
    comb += [(m, m) for m in MODEL_ZOO]
    for pair in list(comb):
        a, b = pair
        pickle1 = os.path.join(args.sim_path, f'{a}.{b}.pkl')
        pickle2 = os.path.join(args.sim_path, f'{b}.{a}.pkl')

        if os.path.exists(pickle1):
            sim_dict = mmcv.load(pickle1)
            print(f'Load {pickle1}')
            all_sim_dict[f'{a}.{b}'] = sim_dict['sim']
            all_sim_dict[f'{b}.{a}'] = sim_dict['sim'].T
        elif os.path.exists(pickle2):
            sim_dict = mmcv.load(pickle2)
            print(f'Load {pickle2}')
            all_sim_dict[f'{a}.{b}'] = sim_dict['sim'].T
            all_sim_dict[f'{b}.{a}'] = sim_dict['sim']
        else:
            AssertionError(f'Either {pickle1} or {pickle2} should exists!')
        # put both key in the dict
    block_sims = Block_Sim(all_sim_dict)
    return block_sims


def main():
    args = parse_args()

    # Load all similarity
    block_sims = get_all_sim(args)
    all_sim = 0

    for k in range(args.trial):
        # Get the initial partition
        block_split_dict = init_partition(args)

        # Get the initial assignment
        all_assignmemt = init_assign(args, block_split_dict, block_sims)

        non_improved = 0
        print(f'Init total Similarity {all_sim}')
        for i in range(args.num_iter):
            # First Step: Re-Partition
            block_split_dict, _ = repartition(
                args, block_split_dict, block_sims, all_assignmemt)

            # Second Step: ReCenter
            all_assignmemt = recenter(
                args, block_split_dict, block_sims, all_assignmemt)

            # Third Step: Reassignment
            all_assignmemt = reassign(
                args, block_split_dict, block_sims, all_assignmemt)

            current_sim = total_cost(all_assignmemt, block_sims)

            if current_sim > all_sim:
                all_sim = current_sim
                best_block_split_dict = block_split_dict
                best_all_assignmemt = all_assignmemt
            else:
                non_improved += 1

            if non_improved > 20:
                print(f'No improvement for {non_improved} iteration')
                print(
                    f'[Trial {k}]: Best Total Similarity {all_sim}, Current Similarity {current_sim}')
                break
    
    print_partition(best_block_split_dict)

    best_all_assignmemt.get_size()
    best_all_assignmemt.print_assignment()
    best_all_assignmemt.save_assignment(
        os.path.join(args.out, f'assignment_hybrid_{args.K}.pkl'))


if __name__ == '__main__':
    main()
