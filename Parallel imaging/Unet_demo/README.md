# U-Net Model for MRI Reconstruction

This directory contains a PyTorch implementation and code for running
pretrained models based on the paper:

[U-Net: Convolutional networks for biomedical image segmentation (O. Ronneberger et al., 2015)](https://doi.org/10.1007/978-3-319-24574-4_28)

which was used as a baseline model in

[fastMRI: An Open Dataset and Benchmarks for Accelerated MRI ({J. Zbontar*, F. Knoll*, A. Sriram*} et al., 2018)](https://arxiv.org/abs/1811.08839)

To start training the model, run:

```bash
python train_unet_demo.py
```

If you run with no arguments, the script will create a `fastmri_dirs.yaml` file
in the root directory that you can use to store paths for your system. You can
also pass options at the command line:

```bash
python train_unet_demo.py \
    --challenge CHALLENGE \
    --data_path DATA \
    --mask_type MASK_TYPE
```

where `CHALLENGE` is either `singlecoil` or `multicoil` and `MASK_TYPE` is
either `random` (for knee) or `equispaced` (for brain). Training logs and
checkpoints are saved in the current working directory by default.

To run the model on test data:

```bash
python train_unet_demo.py \
    --mode test \
    --test_split TESTSPLIT \
    --challenge CHALLENGE \
    --data_path DATA \
    --resume_from_checkpoint MODEL
```

where `MODEL` is the path to the model checkpoint.`TESTSPLIT` should specify
the test split you want to run on - either `test` or `challenge`.

The outputs will be saved to `reconstructions` directory which can be uploaded
for submission.
