# configs/block_mixer/10m_downstream_new/10m_aircraft_256x1_20k_dere_adamw_freeze.py \ 
# configs/block_mixer/10m_downstream_new/10m_caltech101_256x1_20k_dere_adamw_freeze.py \ 
# configs/block_mixer/10m_downstream_new/10m_car_256x1_20k_dere_adamw_freeze.py \
# configs/block_mixer/10m_downstream_new/10m_car_256x1_20k_dere_adamw.py \
# configs/block_mixer/10m_downstream_new/10m_cifar10_256x1_20k_dere_adamw_freeze.py \
# configs/block_mixer/10m_downstream_new/10m_cifar10_256x1_20k_dere_adamw.py \
# configs/block_mixer/10m_downstream_new/10m_cifar100_256x1_20k_dere_adamw_freeze.py \
# configs/block_mixer/10m_downstream_new/10m_cifar100_256x1_20k_dere_adamw.py \
# configs/block_mixer/10m_downstream_new/10m_cifar100_256x1_200e_dere_adamw.py \
# configs/block_mixer/10m_downstream_new/10m_cub_256x1_20k_dere_adamw_freeze.py 
# configs/block_mixer/10m_downstream_new/10m_cub_256x1_20k_dere_adamw.py \
# configs/block_mixer/10m_downstream_new/10m_dtd_256x1_20k_dere_adamw_freeze.py 
# configs/block_mixer/10m_downstream_new/10m_dtd_256x1_20k_dere_adamw.py \
# configs/block_mixer/10m_downstream_new/10m_flower_256x1_20k_dere_adamw_freeze.py 
# configs/block_mixer/10m_downstream_new/10m_flower_256x1_20k_dere_adamw.py \
# configs/block_mixer/10m_downstream_new/10m_pets_256x1_20k_dere_adamw_freeze.py
# configs/block_mixer/20m_downstream_new/20m_cub_256x1_20k_dere_adamw_freeze.py \
# configs/block_mixer/20m_downstream_new/20m_cub_256x1_20k_dere_adamw.py \
# configs/block_mixer/10m_downstream_new/10m_pets_256x1_20k_dere_adamw.py 
# configs/block_mixer/20m_downstream_new/20m_caltech101_256x1_40k_dere_adamw_freeze.py \
# configs/block_mixer/20m_downstream_new/20m_caltech101_256x1_40k_dere_adamw.py \
# configs/block_mixer/20m_downstream_new/20m_cub_256x1_40k_dere_adamw.py \
# configs/block_mixer/20m_downstream_new/20m_pets_256x1_40k_dere_adamw_freeze.py \
# configs/block_mixer/20m_downstream_new/20m_pets_256x1_40k_dere_adamw.py \
# configs/block_mixer/30_downstream_new/30m_caltech101_256x1_40k_dere_adamw_freeze.py \

for config in configs/block_mixer/50m_downstream_new/50m_dtd_64x1_20k_dere_adamw.py \
configs/block_mixer/50m_downstream_new/50m_flower_64x1_20k_dere_adamw.py \
configs/block_mixer/50m_downstream_new/50m_pets_64x1_20k_dere_pretrained_adamw.py
do
python tools/dist_train.py $config 1
done