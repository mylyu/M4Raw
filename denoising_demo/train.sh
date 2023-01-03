#UNET
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name UNET --name std8_15_fastmri_UNET --trainset FastMRITrainSet

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name UNET --name ALL_UNET  --modal ALL

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name UNET --name finetune_UNET  --modal ALL --resume ./M4RawV1.0_experiment/std8_15_fastmri_UNET/snapshot/net_50.pth --lr 1e-5


#NAFNET
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name NAFNET --name std8_15_fastmri_NAFNET --trainset FastMRITrainSet

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name NAFNET --name ALL_NAFNET  --modal ALL

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 11 --loss_l1  --net_name NAFNET --name finetune_NFAET  --modal ALL --resume ./M4RawV1.0_experiment/std8_15_fastmri_NAFNET/snapshot/net_50.pth --lr 1e-5