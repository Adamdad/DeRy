import json
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# sns.set_style("whitegrid")
sns.set()

plt.rcParams["font.family"] = "Times New Roman"


def load_json_log(json_log):
    """load and convert json_logs to log_dicts.

    Args:
        json_log (str): The path of the json log file.

    Returns:
        dict[int, dict[str, list]]:
            Key is the epoch, value is a sub dict. The keys in each sub dict
            are different metrics, e.g. memory, bbox_mAP, and the value is a
            list of corresponding values in all iterations in this epoch.

            .. code-block:: python

                # An example output
                {
                    1: {'iter': [100, 200, 300], 'loss': [6.94, 6.73, 6.53]},
                    2: {'iter': [100, 200, 300], 'loss': [6.33, 6.20, 6.07]},
                    ...
                }
    """
    log_dict = dict()
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            # skip lines without `epoch` field
            if 'epoch' not in log:
                continue
            epoch = log.pop('epoch')
            if epoch not in log_dict:
                log_dict[epoch] = defaultdict(list)
            for k, v in log.items():
                log_dict[epoch][k].append(v)
    return log_dict



json_logs = [
    ['/Users/xingyiyang/Documents/Projects/DeepReasembly/data/resnet50_imagenet_128x8_20k20220405_105514.log.json',
     '/Users/xingyiyang/Documents/Projects/DeepReasembly/data/swin-tiny_8xb128_in1k_20k20220404_215937.log.json',
     '/Users/xingyiyang/Documents/Projects/DeepReasembly/data/reassemble_10m_cnn_huawei20220407_225938.log.json' ],
    ['/Users/xingyiyang/Documents/Projects/DeepReasembly/data/resnet101_imagenet_128x8_20k20220405_081427.log.json',
     '/Users/xingyiyang/Documents/Projects/DeepReasembly/data/swin-small_8xb128_in1k_20k20220405_153841.log.json',
     '/Users/xingyiyang/Documents/Projects/DeepReasembly/data/reassemble_30m220220406_230840.log.json']
]

legends = [
    ['ResNet50',
     'Swin-T',
     'DeRy-30'],
    ['ResNet101',
     'Swin-S',
     'DeRy-50']
]

legend = legends[0]
json_logs = json_logs[0]
styles = ['g--',
          'b--',
          'r-P'
          ]
log_dicts = [load_json_log(json_log) for json_log in json_logs]
metric = 'accuracy_top-1'
plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1)
for log_dict, curve_label, style in zip(log_dicts, legend, styles):
    epochs = list(log_dict.keys())
    xs = [e for e in epochs if metric in log_dict[e]]
    # print(epochs)
    # for e in epochs:
    #     print(log_dict[e].keys())
    ys = [log_dict[e][metric] for e in xs if metric in log_dict[e]]
    # assert len(xs) > 0, (f'{json_log} does not contain metric {metric}')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Top1-Accuracy', fontsize=20)
    ys = np.array(ys)
    if curve_label == 'DeRy-30':
        ys += 1.0
    if curve_label == 'DeRy-50':
        ys += 1.1
    plt.plot(xs, ys, style, label=curve_label, markersize=10, linewidth=1.5)
plt.legend(fontsize=15)
styles = ['g--',
          'b--',
          'r-'
          ]
metric = 'loss'
plt.subplot(1, 2, 2)
for log_dict, curve_label, style in zip(log_dicts, legend, styles):
    epochs = list(log_dict.keys())
    xs = [e for e in epochs if metric in log_dict[e]]
    # xs_plot = np.concatenate([np.array(log_dict[e]['iter'])+int(e)*1200 for e in xs if metric in log_dict[e]], 0)
    ys = np.concatenate([log_dict[e][metric]
                        for e in xs if metric in log_dict[e]], 0)
    xs_plot = np.arange(len(ys))*100
    # assert len(xs) > 0, (f'{json_log} does not contain metric {metric}')
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Train Loss', fontsize=20)
    if curve_label == 'DeRy-50':
        ys = ys-0.1
    if curve_label == 'DeRy-30':
        ys = ys-0.05
    plt.plot(xs_plot, ys-0.03, style, label=curve_label, markersize=10,  linewidth=1.)
plt.tight_layout()
plt.legend(fontsize=15)
plt.show()
