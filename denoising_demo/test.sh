CUDA_VISIBLE_DEVICES=3 python test.py --resume ./M4Raw_exp/ALL_UNET/snapshot/net_100.pth --modal FLAIR --net_name UNET
CUDA_VISIBLE_DEVICES=3 python test.py --resume ./M4Raw_exp/ALL_UNET/snapshot/net_100.pth --modal T1 --net_name UNET
CUDA_VISIBLE_DEVICES=3 python test.py --resume ./M4Raw_exp/ALL_UNET/snapshot/net_100.pth --modal T2 --net_name UNET

CUDA_VISIBLE_DEVICES=3 python test.py --resume ./M4Raw_exp/ALL_NAFNET/snapshot/net_100.pth --modal FLAIR --net_name NAFNET
CUDA_VISIBLE_DEVICES=3 python test.py --resume ./M4Raw_exp/ALL_NAFNET/snapshot/net_100.pth --modal T1 --net_name NAFNET
CUDA_VISIBLE_DEVICES=3 python test.py --resume ./M4Raw_exp/ALL_NAFNET/snapshot/net_100.pth --modal T2 --net_name NAFNET