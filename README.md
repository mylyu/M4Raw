# M4Raw
A Multi-Contrast Multi-Repetition Multi-Channel Raw K-space Dataset for Low-Field MRI Reconstruction
 
# The complete dataset has been uploaded to Zenodo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7523691.svg)](https://doi.org/10.5281/zenodo.7523691)

# Synopsis
We release a new raw k-space dataset M4Raw acquired by the low-field MRI. Currently, it contains multi-channel brain data of 183 subjects each with 18 slices x 3 contrasts (T1w, T2w, and FLAIR). Moreover, each contrast consists of two or three repetitions (a.k.a. NEXs), leading to more than 1k trainable volumes in total, which can be used in various ways by the low-field MRI community.

# Quick Start
1. Download the sample data of one subject (three T2w, three T1w, and two FLAIR scans) from this link.
https://drive.google.com/drive/folders/1RQT9oqoBd0xsevhyKSCxKiLuoEb2erhB?usp=share_link

2. pip install fastmri, which we will use for data visualization in the next step.

3. run the jupter notebook "M4Raw_tutorial.ipynb".


_________________

# TAKE A QUICK LOOK

![image](https://user-images.githubusercontent.com/10205514/203816262-09033556-b5f6-43a3-9dbd-b1dad7e7100a.png)
_________________

** Note: M4Raw is a collaborative research project by Shenzhen Technology University (SZTU), the University of Hong Kong (HKU), Jinan University (JNU), and Shenzhen Samii Medical Center (SSMC).
