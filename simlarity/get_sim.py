import mmcv
import numpy as np
import os
import pandas as pd
# os.system("PYTHONPATH=$PWD")
from blocklize.block_meta import MODEL_BLOCKS, MODEL_PRINT


def main(pickle_file, model1_in, model1_out, model2_in, model2_out):
    sim_dict = mmcv.load(pickle_file)

    model1 = sim_dict['model1']
    model2 = sim_dict['model2']

    sim_map = sim_dict['sim']

    assert (model1['arch'] in MODEL_BLOCKS.keys()
            ), f"{model1['arch']} name must in the MODEL_BLOCKS"
    assert (model2['arch'] in MODEL_BLOCKS.keys()
            ), f"{model2['arch']} name must in the MODEL_BLOCKS"
    model1_block_list = MODEL_BLOCKS[model1['arch']]
    model2_block_list = MODEL_BLOCKS[model2['arch']]

    model1_in_index = model1_block_list.index(model1_in)
    model1_out_index = model1_block_list.index(model1_out)

    model2_in_index = model2_block_list.index(model2_in)
    model2_out_index = model2_block_list.index(model2_out)


    sim = sim_map[model1_in_index, model2_in_index] + \
        sim_map[model1_out_index, model2_out_index]

    return sim


if __name__ == '__main__':
    root = '/Users/xingyiyang/Documents/Projects/DeepReasembly/data/sim_lr/'
    # root = '/Users/xingyiyang/Documents/Projects/DeepReasembly/data/sim_cka/'
    network2layer = {'resnet50': [('layer1.0', 'layer1.2'), ('layer1.2', 'layer2.3'), ('layer2.3', 'layer3.5'), ('layer3.5', 'layer4.2')],
                     'swsl_resnext50_32x4d': [('layer1.0', 'layer1.2'), ('layer1.2', 'layer2.3'), ('layer2.3', 'layer3.5'), ('layer3.5', 'layer4.2')],
                     'resnet101': [('layer1.0', 'layer1.2'), ('layer1.2', 'layer2.3'), ('layer2.3', 'layer3.22'), ('layer3.22', 'layer4.2')],
                     'regnet_y_8gf': [('trunk_output.block1.block1-0', 'trunk_output.block1.block1-1'), ('trunk_output.block1.block1-1', 'trunk_output.block2.block2-3'), ('trunk_output.block2.block2-3', 'trunk_output.block3.block3-9'), ('trunk_output.block3.block3-9', 'trunk_output.block4.block4-0')],
                     'swin_tiny_patch4_window7_224': [('stages.0.blocks.0', 'stages.0.blocks.1'), ('stages.0.blocks.1', 'stages.1.blocks.1'), ('stages.1.blocks.1', 'stages.2.blocks.5'), ('stages.2.blocks.5', 'stages.3.blocks.1')]
                     }
    results = []
    for target in ['swsl_resnext50_32x4d', 'resnet101', 'regnet_y_8gf', 'swin_tiny_patch4_window7_224']:
        file1 = f'resnet50.{target}.pkl'
        file_path1 = os.path.join(root, file1)
        file2 = f'{target}.resnet50.pkl'
        file_path2 = os.path.join(root, file2)
        if os.path.exists(file_path1):
            for i in range(4):
                for j in range(4):
                    sim = main(file_path1,
                               model1_in=network2layer['resnet50'][i][0],
                               model1_out=network2layer['resnet50'][i][1],
                               model2_in=network2layer[target][j][0],
                               model2_out=network2layer[target][j][1])
                    print('resnet50', i, target,j, sim)
                    results.append(['resnet50', i, target,j, sim])
        elif os.path.exists(file_path2):
            for i in range(4):
                for j in range(4):
                    sim = main(file_path2,
                               model2_in=network2layer['resnet50'][i][0],
                               model2_out=network2layer['resnet50'][i][1],
                               model1_in=network2layer[target][j][0],
                               model1_out=network2layer[target][j][1])
                    print('resnet50', i, target,j, sim)
                    results.append(['resnet50', i, target,j, sim])
    df = pd.DataFrame(results)
    df.to_csv('result_lr.csv')
        

