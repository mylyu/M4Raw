import os
# from imageio import imread
from PIL import Image, ImageOps
import numpy as np
import glob
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import sys
sys.path.append('..')
from utils import util

import h5py
import numpy as np
from matplotlib import pyplot as plt
import fastmri
from fastmri.data import transforms as T

def normal(x):
    y = np.zeros_like(x)
    for i in range(y.shape[0]):
        x_min = x[i].min()
        x_max = x[i].max()
        y[i] = (x[i] - x_min)/(x_max-x_min)
    return y
def read_h5(file_name):
    hf = h5py.File(file_name)
    volume_kspace = hf['kspace'][()]
    slice_kspace = volume_kspace
    slice_kspace2 = T.to_tensor(slice_kspace)
    slice_image = fastmri.ifft2c(slice_kspace2)
    slice_image_abs = fastmri.complex_abs(slice_image)
    slice_image_rss = fastmri.rss(slice_image_abs, dim=1)
    slice_image_rss = np.abs(slice_image_rss.numpy())
    slice_image_rss = normal(slice_image_rss)
    return slice_image_rss

class TrainSet(Dataset):
    def __init__(self, args):
        if args.modal == 'T1':
            input_list1 = sorted(glob.glob(os.path.join(args.traindata_root+'/*_T102.h5')))
            input_list2 = [path.replace('_T102.h5','_T101.h5') for path in input_list1]
            input_list3 = [path.replace('_T102.h5','_T103.h5') for path in input_list1]
            all = [input_list1,input_list2,input_list3]
        elif args.modal == 'T2':
            input_list1 = sorted(glob.glob(os.path.join(args.traindata_root+'/*_T202.h5')))
            input_list2 = [path.replace('_T202.h5','_T201.h5') for path in input_list1]
            input_list3 = [path.replace('_T202.h5','_T203.h5') for path in input_list1]
            all = [input_list1,input_list2,input_list3]
        elif args.modal == 'FLAIR':
            input_list1 = sorted(glob.glob(os.path.join(args.traindata_root+'/*_FLAIR02.h5')))
            input_list2 = [path.replace('_FLAIR02.h5','_FLAIR01.h5') for path in input_list1]
            all = [input_list1,input_list2]
        self.all = all
        self.images = np.zeros([len(input_list1),len(all), 18, 256, 256])
        print('TrainSet loading...')
        for i in range(len(self.all)):
            for j,path in enumerate(all[i]):
                self.images[j][i] = read_h5(path)
        self.labels = np.mean(self.images,axis=1)
        print('over load')
        
        self.images = self.images.transpose(0,2,1,3,4).reshape(-1,len(all),256,256)
        self.labels = self.labels.reshape(-1,1,256,256)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        choisce = np.random.choice([i for i in range(len(self.all))],1)
        images = images[choisce]
        sample = {'images':images,
                  'labels':labels
                 }
        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32)
            sample[key] = torch.from_numpy(sample[key]).float()

        return sample
class TestSet(Dataset):
    def __init__(self, args):
        if args.modal == 'T1':
            input_list1 = sorted(glob.glob(os.path.join(args.testdata_root+'/*_T102.h5')))
            input_list2 = [path.replace('_T102.h5','_T101.h5') for path in input_list1]
            input_list3 = [path.replace('_T102.h5','_T103.h5') for path in input_list1]
            all = [input_list1,input_list2,input_list3]
        elif args.modal == 'T2':
            input_list1 = sorted(glob.glob(os.path.join(args.testdata_root+'/*_T202.h5')))
            input_list2 = [path.replace('_T202.h5','_T201.h5') for path in input_list1]
            input_list3 = [path.replace('_T202.h5','_T203.h5') for path in input_list1]
            all = [input_list1,input_list2,input_list3]
        elif args.modal == 'FLAIR':
            input_list1 = sorted(glob.glob(os.path.join(args.testdata_root+'/*_FLAIR02.h5')))
            input_list2 = [path.replace('_FLAIR02.h5','_FLAIR01.h5') for path in input_list1]
            all = [input_list1,input_list2]
        self.all = all
        self.images = np.zeros([len(input_list1),len(all), 18, 256, 256])
        print('TestSet loading...')
        for i in range(len(self.all)):
            for j,path in enumerate(all[i]):
                self.images[j][i] = read_h5(path)
        self.labels = np.mean(self.images,axis=1)
        print('over load')
        
        self.images = self.images.transpose(0,2,1,3,4).reshape(-1,len(all),256,256)
        self.labels = self.labels.reshape(-1,1,256,256)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        choisce = np.random.choice([0],1)
        images = images[choisce]
        sample = {'images':images,
                  'labels':labels
                 }
        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32)
            sample[key] = torch.from_numpy(sample[key]).float()

        return sample

class FastMRITrainSet(Dataset):
    def __init__(self, args):
        p_t2 = glob.glob('/data2/fastmri/png/train/*AXT2*') + glob.glob('/data2/fastmri/png/val/*AXT2*')
        p_flair = glob.glob('/data2/fastmri/png/train/*AXFLAIR*') + glob.glob('/data2/fastmri/png/val/*AXFLAIR*')
        p_t1 = glob.glob('/data2/fastmri/png/train/*AXT1_*') + glob.glob('/data2/fastmri/png/val/*AXT1_*')
        p_t1post = glob.glob('/data2/fastmri/png/train/*AXT1POST*') + glob.glob('/data2/fastmri/png/val/*AXT1POST*')
        p_t1pre = glob.glob('/data2/fastmri/png/train/*AXT1PRE*') + glob.glob('/data2/fastmri/png/val/*AXT1PRE*')

        # 0.8 / 0.1 / 0.1
        random.Random(0).shuffle(sorted(p_t2))
        random.Random(0).shuffle(sorted(p_flair))
        random.Random(0).shuffle(sorted(p_t1))
        random.Random(0).shuffle(sorted(p_t1post))
        random.Random(0).shuffle(sorted(p_t1pre))

        # database = p_t1[:272]+p_t1post[:988] + p_t1pre[:261] + p_t2[:2794] + p_flair[:360]
        database = p_t1+p_t1post + p_t1pre + p_t2 + p_flair
        
        self.paths = []
        for patient in database:
            paths = sorted(glob.glob(patient+'/*.png'))
            for path in paths[:10]:
                self.paths.append(path)
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        HR = cv2.imread(self.paths[idx])
        HR = cv2.resize(HR,(256,256))[:,:,0][:,:,None]
        
        h, w, _ = HR.shape
        noise = np.random.normal(0, np.random.randint(8,15), (h, w,1))
        LR = HR + noise
        LR = np.clip(LR,0,255)
        
        sample = {'images':LR,
                  'labels':HR
                 }
        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        return sample
    
class FastMRITestSet(Dataset):
    def __init__(self, args):
        p_t2 = glob.glob('/data2/fastmri/png/train/*AXT2*') + glob.glob('/data2/fastmri/png/val/*AXT2*')
        p_flair = glob.glob('/data2/fastmri/png/train/*AXFLAIR*') + glob.glob('/data2/fastmri/png/val/*AXFLAIR*')
        p_t1 = glob.glob('/data2/fastmri/png/train/*AXT1_*') + glob.glob('/data2/fastmri/png/val/*AXT1_*')
        p_t1post = glob.glob('/data2/fastmri/png/train/*AXT1POST*') + glob.glob('/data2/fastmri/png/val/*AXT1POST*')
        p_t1pre = glob.glob('/data2/fastmri/png/train/*AXT1PRE*') + glob.glob('/data2/fastmri/png/val/*AXT1PRE*')

        # 0.8 / 0.1 / 0.1
        random.Random(0).shuffle(sorted(p_t2))
        random.Random(0).shuffle(sorted(p_flair))
        random.Random(0).shuffle(sorted(p_t1))
        random.Random(0).shuffle(sorted(p_t1post))
        random.Random(0).shuffle(sorted(p_t1pre))

        val = p_t1[272:]+p_t1post[988:] + p_t1pre[261:] + p_t2[2794:] + p_flair[360:]
        
        self.paths = []
        for patient in val:
            paths = sorted(glob.glob(patient+'/*.png'))
            for path in paths[:10]:
                self.paths.append(path)
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        HR = cv2.imread(self.paths[idx])
        HR = cv2.resize(HR,(256,256))[:,:,0][:,:,None]
        
        h, w, _ = HR.shape
        noise = np.random.normal(0, np.random.randint(8,12), (h, w,1))
        LR = HR + noise
        LR = np.clip(LR,0,255)
        
        sample = {'images':LR,
                  'labels':HR
                 }
        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32) / 255.
            sample[key] = torch.from_numpy(sample[key]).permute(2, 0, 1).float()

        return sample
