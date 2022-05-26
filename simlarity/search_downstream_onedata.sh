#  airplane caltech101 cars caltech101 cifar10 cifar100 dtd cub flower pets
for data in cars
do
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" python simlarity/zeroshot_nas_hybird.py \
--path simlarity/out/assignment/assignment_hybrid_4.pkl \
--zero_proxy naswot \
--num_batch 5 \
--batch_size 32 \
--trial 20 \
--C 10.0 \
--flop_C 3.0 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" python simlarity/zeroshot_nas_hybird.py \
--path simlarity/out/assignment/assignment_hybrid_4.pkl \
--zero_proxy naswot \
--num_batch 5 \
--batch_size 32 \
--trial 20 \
--C 20.0 \
--minC 10.0 \
--flop_C 6.0 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" python simlarity/zeroshot_nas_hybird.py \
--path simlarity/out/assignment/assignment_hybrid_4.pkl \
--zero_proxy naswot \
--num_batch 5 \
--batch_size 32 \
--trial 20 \
--C 30.0 \
--flop_C 10.0 \
--minC 20.0 \
--minflop_C 3.0 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" python simlarity/zeroshot_nas_hybird.py \
--path simlarity/out/assignment/assignment_hybrid_4.pkl \
--zero_proxy naswot \
--num_batch 5 \
--batch_size 32 \
--trial 20 \
--C 50.0 \
--flop_C 15.0 \
--minC 40.0 \
--minflop_C 5.0 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py

CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" python simlarity/zeroshot_nas_hybird.py \
--path simlarity/out/assignment/assignment_hybrid_4.pkl \
--zero_proxy naswot \
--num_batch 5 \
--batch_size 32 \
--trial 20 \
--C 90.0 \
--flop_C 20.0 \
--minC 70.0 \
--minflop_C 10.0 \
--data_config configs/_base_/datasets/"$data"_bs64_2.py
done