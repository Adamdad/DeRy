# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import time
import warnings
from tkinter.messagebox import NO

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from blocklize import MODEL_BLOCKS
from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.apis.test import collect_results_cpu, collect_results_gpu
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from timm import models as tm
from torch import nn

# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model


class GetFeatureHook:
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.feature = []

    def hook_fn(self, module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        self.in_size = input.shape[1:]
        self.out_size = output.shape[1:]
        output = F.adaptive_avg_pool2d(output, (1, 1)).detach().cpu()
        self.feature.append(output)

    def concat(self):
        self.feature = torch.cat(self.feature, dim=0)

    def flash_mem(self):
        del self.feature
        self.feature = []

    def close(self):
        self.hook.remove()


class GetFeatureHook_transformer(GetFeatureHook):
    def __init__(self, module):
        super().__init__(module)
        print('Transformer Get Feature Hook')

    def hook_fn(self, module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        self.in_size = input.shape[1:]
        if isinstance(output, tuple):
            output = output[0]
        self.out_size = output.shape[1:]
        output = F.adaptive_avg_pool1d(
            output.transpose(1, 2), 1).detach().cpu()
        self.feature.append(output)


class SetFeatureHook:
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_pre_hook(self.hook_fn)

    def set_feat(self, feat):
        self.feat = feat
        self.batch_id = 0
        self.valid = True

    def hook_fn(self, module, input):
        # Get the current batch of data
        if isinstance(input, tuple):
            input = input[0]
        bs, out_size, h, w = input.size()

        batch_feat = self.feat.to(input.device)
        bs, in_size, new_h, new_w = batch_feat.size()

        # if the size mismatch great than 2^3
        if np.log2(h/new_h) >= 3:
            new_input = input
            self.valid = False
        else:
            # Create the orthognal transform

            # transform = torch.zeros((out_size, in_size))
            # transform.fill_diagonal_(1.0)

            transform = torch.empty((out_size, in_size))
            torch.nn.init.orthogonal_(transform)
            # add two dumpy dimension
            transform = transform.unsqueeze(-1).unsqueeze(-1).to(input.device)

            new_input = F.conv2d(batch_feat, transform, stride=1, padding=0)

        return new_input

    def close(self):
        self.hook.remove()


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint',  default='', help='checkpoint file')
    parser.add_argument('--out', default='', help='output result file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main_block():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # assert args.metrics or args.out, \
    #     'Please specify at least one of output path and evaluation metrics.'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=True)

    assert cfg.model.train_cfg.model_name in MODEL_BLOCKS.keys(), \
        "Model name must in the MODEL_BLOCKS"

    train_strategy = cfg.model.train_cfg.get('train_strategy', None)

    block_list = MODEL_BLOCKS[cfg.model.train_cfg.model_name]
    feature_layers = dict()
    feature_layers_names = []
    # build the model and load checkpoint
    if cfg.model.train_cfg.model_name.startswith('vit') or cfg.model.train_cfg.model_name.startswith('swin'):
        hook_class = GetFeatureHook_transformer
    else:
        hook_class = GetFeatureHook

    model = build_classifier(cfg.model)
    # model.init_weights()
    # exit()
    ckp_path = cfg.get('load_from', None)
    if ckp_path is not None:
        print(f'Loading from {ckp_path}')
        load_checkpoint(model, ckp_path, revise_keys=[(r'^module\.backbone\.', ''),
                                                      (r'^backbone\.', '')])

    for name, module in model.named_modules():
        for blk_name in block_list:
            if name.endswith(blk_name):
                feature_layers_names.append(name)
                feature_layers[name] = hook_class(module)
                break

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    if not distributed:
        if args.device == 'cpu':
            model = model.cpu()
        else:
            model = MMDataParallel(model, device_ids=[0])
        # model.CLASSES = CLASSES
        show_kwargs = {} if args.show_options is None else args.show_options
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

        # collect results from all ranks
        for layer_name in feature_layers_names:
            layer = feature_layers[layer_name]
            if args.gpu_collect:
                layer.feature = collect_results_gpu(
                    layer.feature, len(dataset))
            else:
                layer.feature = collect_results_cpu(
                    layer.feature, len(dataset), args.tmpdir)

    rank, _ = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(feature_layers_names))
        feat = dict()
        size = dict()
        for layer_name in feature_layers_names:
            layer = feature_layers[layer_name]
            layer.concat()
            prog_bar.update()
            num = layer.feature.size(0)
            size[layer_name] = [layer.in_size, layer.out_size]
            feat[layer_name] = layer.feature.view(num, -1)
        if train_strategy:
            results = dict(model_name=cfg.model.train_cfg.model_name,
                           train_strategy=train_strategy,
                           size=size,
                           feat=feat)
        else:
            results = dict(model_name=cfg.model.train_cfg.model_name,
                           size=size,
                           feat=feat)
        # mmcv.dump(results, args.out)
        torch.save(results, args.out)


if __name__ == '__main__':
    main_block()
