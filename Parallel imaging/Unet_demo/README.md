# U-Net Model for MRI Reconstruction

This directory contains a PyTorch implementation and code for running
pretrained models based on the paper:

[U-Net: Convolutional networks for biomedical image segmentation (O. Ronneberger et al., 2015)](https://doi.org/10.1007/978-3-319-24574-4_28)

which was used as a baseline model in

[fastMRI: An Open Dataset and Benchmarks for Accelerated MRI ({J. Zbontar*, F. Knoll*, A. Sriram*} et al., 2018)](https://arxiv.org/abs/1811.08839)

The first thing you need to do is install fastMRI in your Python environment:

```bash
pip install fastMRI
```

Then the M4Raw data set is divided into multicoil_train, multicoil_val and multicoil_test:

```
data_root_path
---multicoil_train
------***.h5
---multicoil_val
------***.h5
---multicoil_test
------***.h5
```

To start training demo the model, run:

```bash
sh train.sh
```

You need to modify the contents of the.sh file as required, such as the number of GPUs and the root directory for storing data.
