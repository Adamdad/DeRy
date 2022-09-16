# Pretrained checkpoint that are used for reassembly

| Architecture  | Discription  | From Repo |  Url |  
|---|---|---|---|
|  ResNet50 |  Trained on ImageNet with SimCLR loss with 200 epoch | [mmselfsup](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/model_zoo.md)  | [checkpoint](https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_20220428-46ef6bb9.pth)  |  
|  ResNet50 |  Trained on ImageNet with BYOL loss with 200 epoch | [mmselfsup](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/model_zoo.md)  | [checkpoint](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k_20220225-5c8b2c2e.pth)  |  
|  ResNet50 |  Trained on ImageNet with MoCov2 loss with 800 epoch  |  [mocov2](https://github.com/facebookresearch/moco) |  [checkpoint](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar) |  
|  ResNet50 |  Supervised Trained on iNat2021  |  [newt](https://github.com/visipedia/newt/tree/main/benchmark) |  [checkpoint](https://cornell.box.com/s/bnyhq5lwobu6fgjrub44zle0pyjijbmw) |  
| ResNet50  | Trained on the datasets pc-nih-rsna-siim-vin at a 512x512 resolution.  | [torchxray](https://github.com/mlmed/torchxrayvision)  | [checkpoint](https://github.com/mlmed/torchxrayvision/releases/download/v1/pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt)  |  
|  ViT-base |  Trained on ImageNet with Mask Reconstruction Loss  |  [mae](https://github.com/facebookresearch/mae) |  [checkpoint](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) |  
|  ViT-base |  Trained on ImageNet with MoCov3 Loss  |  [mocov3](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md) |  [checkpoint](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar) |  
|  ViT-small |  Trained on ImageNet with MoCov3 Loss  |  [mocov3](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md) |  [checkpoint](https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar) |  