_base_ = [
    '../_base_/models/swin_transformer/small_224.py',
    '../_base_/datasets/imagenet_bs64.py',
    '../_base_/default_runtime.py'
]

model = dict(
    train_cfg=dict(model_name='swin_small_patch4_window7_224'),
)


data = dict(
    samples_per_gpu=64,
    val=dict(ann_file=None),
    test=dict(ann_file=None))

load_from = 'https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_small_patch4_window7_224-cc7a01c9.pth'