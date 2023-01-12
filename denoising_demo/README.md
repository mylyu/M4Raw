# Train
## Train in all slice with NAFNET model
```python
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 51 --traindata_root /data0/M4RawV1.0/multicoil_train/ --loss_l1 --net_name NAFNET --name random_init_NAFNET --lr 1e-4 --modal ALL
```

## Train in all slice with UNET model
```python
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 51 --traindata_root /data0/M4RawV1.0/multicoil_train/ --loss_l1 --net_name UNET --name random_init_NAFNET --lr 1e-4 --modal ALL 
```


# Inference
## For Inference in T1 modal with NAFNET model

```python
python test.py --net_name NAFNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal T1
```

## For Inference in T2 modal with NAFNET model
```python
python test.py --net_name NAFNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal T2
```
## For Inference in FLAIR modal with NAFNET model
```python
python test.py --net_name NAFNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal FLAIR
```
## For Inference in T1 modal with UNET model
```python
python test.py --net_name UNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal T1
```
## For Inference in T2 modal with UNET model
```python
python test.py --net_name UNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal T2
```
## For Inference in FLAIR modal with UNET model
```python
python test.py --net_name UNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal FLAIR
```
