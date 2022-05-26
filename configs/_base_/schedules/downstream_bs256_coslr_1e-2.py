# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
runner = dict(
    type='IterBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_iters=20000) # Total number of iterations. For EpochBasedRunner use max_epochs
