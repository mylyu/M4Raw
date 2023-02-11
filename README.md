# M4Raw
M4Raw: A multi-contrast, multi-repetition, multi-channel MRI k-space dataset for low-field MRI research

# The complete dataset has been uploaded to Zenodo
https://doi.org/10.5281/zenodo.7523691

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7523691.svg)](https://doi.org/10.5281/zenodo.7523691)

It can also be downloaded from BaiduYun
Link：https://pan.baidu.com/s/10MikOaLpsF86RP4wYIbNhw?pwd=kizp 
passcode：kizp 

# Abstract
Recently, low-field magnetic resonance imaging (MRI) has gained renewed interest to promote MRI accessibility and affordability worldwide. The presented M4Raw dataset aims to facilitate methodology development and reproducible research in this field. The dataset comprises multi-channel brain k-space data collected from 183 healthy volunteers using a 0.3 Tesla whole-body MRI system, and includes T1-weighted, T2-weighted, and fluid attenuated inversion recovery (FLAIR) images with in-plane resolution of ~1.2 mm and through-plane resolution of 5 mm. Importantly, each contrast contains multiple repetitions, which can be used individually or to form multi-repetition averaged images. After excluding motion-corrupted data, the partitioned training and validation subsets contain 1024 and 240 volumes, respectively. To demonstrate the potential utility of this dataset, we trained deep learning models for image denoising and parallel imaging tasks and compared their performance with traditional reconstruction methods. This M4Raw dataset will be valuable for the development of advanced data-driven methods specifically for low-field MRI. It can also serve as a benchmark dataset for general MRI reconstruction algorithms.

<img src="https://user-images.githubusercontent.com/10205514/218274571-a69e84ef-6b02-46fc-9457-68b0cda0d96b.png" height="400" />

# Quick Start
1. Download the sample data of one subject (three T2w scans, three T1w scans, and two FLAIR scans) from this link.
https://drive.google.com/drive/folders/1RQT9oqoBd0xsevhyKSCxKiLuoEb2erhB?usp=share_link

2. pip install fastmri, which we will use for data visualization in the next step.

3. run the jupter notebook "M4Raw_tutorial.ipynb".


_________________

# TAKE A QUICK LOOK
<img src="https://user-images.githubusercontent.com/10205514/211978406-a4fc010e-b3f9-4d65-bf97-ec2abc8db725.png" height="400" />


# News

**2023/02/06**: The M4Raw pretrained model weights have been uploaded to github (see Release).

**2023/01/11**: The complete M4Raw dataset has been released to https://zenodo.org/record/7523691 .

_________________

** Note: M4Raw is a collaborative research project by Shenzhen Technology University (SZTU), the University of Hong Kong (HKU), Jinan University (JNU), and Shenzhen Samii Medical Center (SSMC).
