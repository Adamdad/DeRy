# Deep Model Reassembly
## Introduction
This repository contains the offical implementation for our paper

**Deep Model Reassembly**

*Submitted to NeurIPS 2022*

In this work, we explore a novel knowledge-transfer task, termed as Deep Model Reassembly (*DeRy*), for general-purpose model reuse. *DeRy* first dissect each model into distinctive building blocks, and then selectively reassemble the derived blocks to produce customized networks under both the hardware resource and performance constraints.

![pipeline](assets/pipeline.png)

## File Orgnization

    blocklize/block_meta.py         [Meta Information & Node Defnition]

    similarity/
        get_rep.py                  [Compute and save the feature embeddings]
        get_sim.py                  [Compute representation similarity given the saved features]
        partition.py                [Network partition by cover set problem]
        zeroshot_reassembly.py      [Network reassembly by solving integer program]

    configs/
        compute_sim/                [Model configs in the model zoo to 
                                     compute the feature similarity]
        dery/XXX/$ModelSize_$DataSet_$BatchSize_$TrainTime_dery_$Optimizor.py
                                    [Config files for transfer experiments]

    mmcls/
        datasets/                   [Dataset definitions]
        models/backbones/dery.py    [DeRy backbone definition]

    third_package/timm              [Modified timm package]

    

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
The *DeRy* consists of 3 steps
1. [**Model Zoo Preparation**] Compute the model feature embeddings and representation similarity. We first write model configuration and its weight path, and run
            
        PYTHONPATH="$PWD" python get_rep.py $Config_file

    The feature embeddings will be saved in *.pth* files in the same $Feat_dictionary. We then load them and compute the feature similarity.

        PYTHONPATH="$PWD" python compute_sim.py /
        --feat_path $Feat_dictionary /
        --sim_func $Similarity_function [cka, rbf_cka, lr]

    We also need to compute the feature size (input-output feature dimensions). It can be done by running

        PYTHONPATH="$PWD" python count_inout_size.py /
        --root $Feat_dictionary

2. [**Network Partition**] Solve the cover set optimization to get the network partition. The results is an assignment file in *.pkl*.

        python partition.py /
        --sim_path $Feat_similarity_path /
        --K        $Num_partition /
        --trial    $Num_repeat_runs /
        --eps      $Size_ratio_each_block /
        --num_iter $Maximum_num_iter_eachrun

3. [**Reassemby**] Reassemble the partitioned building blocks into a full model, by solving a integer program. The results are a series of model configs in *.py*.

        PYTHONPATH="$PWD" python zeroshot_reassembly.py \
        --path          $Block_partition_file [Saved in the partition step] \
        --C             $Maximum_parameter_num \
        --minC          $Minimum_parameter_num \
        --flop_C        $Maximum_FLOPs_num \
        --minflop_C     $Minimum_FLOPs_num \
        --num_batch     $Number_batch_average_to_compute_score \
        --batch_size    $Number_sample_each_batch \
        --trial         $Search_time \
        --zero_proxy    $Type_train_free_proxy [Default NASWOT] \
        --data_config   $Config_target_data

4. [**Fune-tuning**] Train the reassembled model on target data. You may refers to [mmclassification](https://github.com/open-mmlab/mmclassification) for the model training.

 

