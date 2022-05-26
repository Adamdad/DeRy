import json
import os
import torch

from blocklize.block_meta import MODEL_BLOCKS

root = '/home/yangxingyi/InfoDrop/simlarity/out/cka_pth'

PYTHS = os.listdir(root)
PYTHS = [os.path.join(root, p) for p in PYTHS if p.endswith('.pth')]


def main():
    # args = parse_args()
    # PYTHS = [os.path.join(args.pickle_path, f) for f in os.listdir(args.pickle_path) if f.endswith('pkl')]

    MODEL_INOUT_SHAPE={}
    for pickle in PYTHS:
        data = torch.load(pickle)
        name = data['model_name']
        arch = name
        if 'train_strategy' in data.keys():
            name += data['train_strategy']
        print(f'Get InOut size for {name}')
        
        assert name in MODEL_BLOCKS.keys(), f'{name} must be a valid mode in MODEL_BLOCKS'
        MODEL_INOUT_SHAPE[name]= dict(in_size=dict(), out_size=dict())
        for key in data['size'].keys():
            for layer in MODEL_BLOCKS[name]:
                if key.endswith(layer):
                    in_size, out_size = data['size'][key]
                    MODEL_INOUT_SHAPE[name]['in_size'][layer] = tuple(in_size)
                    MODEL_INOUT_SHAPE[name]['out_size'][layer] = tuple(out_size)

        del data
    
    with open('tools/MODEL_INOUT_SHAPE.json', 'w') as fp:
        json.dump(MODEL_INOUT_SHAPE, fp, indent=2)
        

if __name__ == '__main__':
    main()
    