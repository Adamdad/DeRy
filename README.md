# Deep Model Reassembly
## Introduction
This repository contains the offical implementation for our paper

**Deep Model Reassembly**

*Submitted to NeurIPS 2022*

In this work, we explore a novel knowledge-transfer task, termed as Deep Model Reassembly (*DeRy*), for general-purpose model reuse. *DeRy* first dissect each model into distinctive building blocks, and then selectively reassemble the derived blocks to produce customized networks under both the hardware resource and performance constraints.

![pipeline](assets/pipeline.png)

## File Orgnization

    blocklize/block_meta.py [Meta Information & Node Defnition]
    

## Installation
The model training part is based on [mmclassification](https://github.com/open-mmlab/mmclassification). 

    conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
    conda activate open-mmlab
    pip3 install openmim
    mim install mmcv-full
    git clone https://github.com/open-mmlab/mmclassification.git
    cd mmclassification
    pip3 install -e .


## Getting Started