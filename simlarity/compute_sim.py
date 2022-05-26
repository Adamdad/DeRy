import argparse
import os
from importlib.resources import path
from itertools import combinations

import mmcv
import torch

from simlarity.compare_functions import SIM_FUNC


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('--feat_path', type=str, help='a path that contains all pickle files that contains the features')
    parser.add_argument('--out', default='', help='output result file')
    parser.add_argument('--sim_func', default='cka', choices=['cka', 'rbf_cka', 'lr'], help='output result file')
    args = parser.parse_args()
    return args

def make_meta(args):
    if os.path.exists(args.name_list):
        print(f'Name List Exists {args.name_list}')
        meta = mmcv.load(args.name_dict)
    else:
        print('Name List Not Exists')
        paths = [os.path.join(args.feat_path, f) for f in os.listdir(args.feat_path) if f.endswith('pkl')]
        meta = []
        for pickle in paths:
            data = mmcv.load(pickle)
            model_meta = dict(model_name=data['model_name'],model_path=pickle)
            if 'train_strategy' in data.keys():
                model_meta['train_strategy'] = data['train_strategy']
            meta.append(model_meta)
        mmcv.dump(meta, args.name_list)
        print(f'Name List Created {args.name_list}')
    return meta
def main():
    args = parse_args()
    # PYTHS = [os.path.join(args.pickle_path, f) for f in os.listdir(args.pickle_path) if f.endswith('pkl')]
    PYTHS = os.listdir(args.feat_path)
    PYTHS = [os.path.join(args.feat_path, p) for p in PYTHS]
    
    pkls_comb = list(combinations(PYTHS, 2))
    pkls_comb += [(pkl, pkl) for pkl in PYTHS]
    
    for pickle1, pickle2 in reversed(pkls_comb):
        # data1 = mmcv.load(pickle1)
        # data2 = mmcv.load(pickle2)
        data1 = torch.load(pickle1)
        data2 = torch.load(pickle2)
        name1 = data1['model_name']
        name2 = data2['model_name']
        arch1 = name1
        arch2 = name2
        if 'train_strategy' in data1.keys():
            name1 += data1['train_strategy']
        if 'train_strategy' in data2.keys():
            name2 += data2['train_strategy']
        
        save_path = os.path.join(args.out, f'{name1}.{name2}.pkl')
        if os.path.exists(save_path):
            print(f'{save_path} already exists.')
            continue
        
        print(f'Computing {name1}.{name2} similarity')
        sim = SIM_FUNC[args.sim_func](data1, data2, bs=2048)
        print('\n')
        print('*'*40)
        print(f"Saving {name1}.{name2} similarity to {save_path}")
        results = dict(sim=sim, 
                        model1=dict(arch = arch1, model_name=name1),
                        model2=dict(arch = arch2, model_name=name2))
        mmcv.dump(results, save_path)
        del data1
        del data2
        

if __name__ == '__main__':
    main()
    