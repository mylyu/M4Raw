# End-to-End Variational Networks for Accelerated MRI Reconstruction Model

This directory contains a PyTorch implementation and code for running
pretrained models for reproducing the paper:

[End-to-End Variational Networks for Accelerated MRI Reconstruction ({A. Sriram*, J. Zbontar*} et al., 2020)][e2evarnet]

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

## Citing

If you use this this code in your research, please cite the corresponding
paper:

```BibTeX
@inproceedings{sriram2020endtoend,
    title={End-to-End Variational Networks for Accelerated MRI Reconstruction},
    author={Anuroop Sriram and Jure Zbontar and Tullie Murrell and Aaron Defazio and C. Lawrence Zitnick and Nafissa Yakubova and Florian Knoll and Patricia Johnson},
    year={2020},
    eprint={2004.06688},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

## Implementation Notes

There are a few differences between this implementation and the
[paper][e2evarnet].

- The paper model used a fixed number of center lines, whereas this model uses
the `center_fractions` variable that might change depending on image size.
- The paper model was trained separately on 4x and 8x, whereas this model
trains on both of them together.

These differences have been left partly for backwards compatibility and partly
due to the number of areas in the code base that would have go be tweaked and
tested to get them working.

[leadlink]: https://fastmri.org/leaderboards/
[e2evarnet]: https://doi.org/10.1007/978-3-030-59713-9_7
