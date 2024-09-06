#UNET
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name UNET --name std8_15_fastmri_UNET --trainset FastMRITrainSet

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name UNET --name ALL_UNET  --modal ALL

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name UNET --name finetune_UNET  --modal ALL --resume ./M4RawV1.0_experiment/std8_15_fastmri_UNET/snapshot/net_50.pth --lr 1e-5


#NAFNET
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name NAFNET --name std8_15_fastmri_NAFNET --trainset FastMRITrainSet

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name NAFNET --name ALL_NAFNET  --modal ALL

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name NAFNET --name finetune_NFAET  --modal ALL --resume ./M4RawV1.0_experiment/std8_15_fastmri_NAFNET/snapshot/net_50.pth --lr 1e-5

python train.py --launcher pytorch --max_iter 100 --loss_l1 --net_name SMPUNET --name random_init_SMPUNET --lr 1e-4 --modal ALL --gpu_ids 0 --launcher none --batch_size 16

# New models
CUDA_VISIBLE_DEVICES=2,5 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch   --max_iter 100 --loss_l1 --net_name SMPUNET --name HXH_pretrain_SMPUNET_formal2 --lr 1e-3 --modal ALL --val_modal ALL --batch_size 64  --test_freq 5 --dataset PNG --trainset PNGDataset --testset PNGDataset --traindata_root /data2/hxh/SSL/datasets/All_Datasets/brain_all_data_21W/  --testdata_root /data2/hxh/SSL/datasets/All_Datasets/brain_all_data_21W/ --noise_std_low 0.03 --noise_std_high 0.08 --save_epoch_freq 10