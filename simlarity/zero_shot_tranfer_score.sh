for data in airplane caltech101 cars caltech101 cifar10 cifar100 dtd cub flower pets
do
echo configs/_base_/datasets/"$data"_bs64_2.py
echo simlarity/out/nasi_$data.json
PYTHONPATH="$PWD" python simlarity/zeroshot_nas_modelzoo.py \
--out simlarity/out/ntk_$data.json \
--zero_proxy ntk \
--num_batch 10 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py
done

for data in airplane caltech101 cars caltech101 cifar10 cifar100 dtd cub flower pets
do
echo configs/_base_/datasets/"$data"_bs64_2.py
echo simlarity/out/nasi_$data.json
CUDA_VISIBLE_DEVICES=2  PYTHONPATH="$PWD" python simlarity/zeroshot_nas_modelzoo.py \
--out simlarity/out/naswot_$data.json \
--zero_proxy naswot \
--num_batch 10 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py
done

for data in airplane caltech101 cars caltech101 cifar10 cifar100 dtd cub flower pets
do
echo configs/_base_/datasets/"$data"_bs64_2.py
echo simlarity/out/nasi_$data.json
PYTHONPATH="$PWD" python simlarity/zeroshot_nas_modelzoo.py \
--out simlarity/out/fisher_$data.json \
--zero_proxy fisher \
--num_batch 10 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py
done

for data in airplane caltech101 cars caltech101 cifar10 cifar100 dtd cub flower pets
do
echo configs/_base_/datasets/"$data"_bs64_2.py
echo simlarity/out/nasi_$data.json
PYTHONPATH="$PWD" python simlarity/zeroshot_nas_modelzoo.py \
--out simlarity/out/snip_$data.json \
--zero_proxy snip \
--num_batch 10 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py
done

for data in airplane caltech101 cars caltech101 cifar10 cifar100 dtd cub flower pets
do
echo configs/_base_/datasets/"$data"_bs64_2.py
echo simlarity/out/nasi_$data.json
PYTHONPATH="$PWD" python simlarity/zeroshot_nas_modelzoo.py \
--out simlarity/out/synflow_$data.json \
--zero_proxy synflow \
--num_batch 10 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py
done


for data in airplane caltech101 cars caltech101 cifar10 cifar100 dtd cub flower pets
do
echo configs/_base_/datasets/"$data"_bs64_2.py
echo simlarity/out/nasi_$data.json
PYTHONPATH="$PWD" python simlarity/zeroshot_nas_modelzoo.py \
--out simlarity/out/grad_norm_$data.json \
--zero_proxy grad_norm \
--num_batch 10 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py
done