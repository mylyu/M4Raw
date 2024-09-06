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
        all_list = []
        for i in range(1,10):
            input_list = sorted(glob.glob(os.path.join(args.traindata_root+'/*_{}0{}.h5'.format(args.modal,i))))
            if len(input_list)>0:
                print('found {}, repetition #{}: {} subjects'.format(args.modal, i, len(input_list)))
                all_list.append(input_list)
        input_list1 = all_list[0]
        self.all = all_list
        #probe the image dimensions
        example_image = read_h5(input_list1[0])
        print(f'the image dimension is {example_image.shape}')
        self.images = np.zeros([len(input_list1),len(all_list), 
                                example_image.shape[-3], 
                                example_image.shape[-2],
                                example_image.shape[-1]])
        print('TrainSet loading...')
        for i in range(len(self.all)):
            for j,path in enumerate(all_list[i]):
                self.images[j][i] = read_h5(path)
        self.labels = np.mean(self.images,axis=1)
        print('Finished loading')
        
        self.images = self.images.transpose(0,2,1,3,4).reshape(-1,len(all_list),example_image.shape[-2],example_image.shape[-1])
        self.labels = self.labels.reshape(-1,1,example_image.shape[-2],example_image.shape[-1])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        rep_choice = np.random.choice([i for i in range(len(self.all))],1)
        images = images[rep_choice]
        sample = {'images':images,
                  'labels':labels
                 }
        for key in sample.keys():
            sample[key] = sample[key].astype(np.float32)
            sample[key] = torch.from_numpy(sample[key]).float()

        return sample
    
class TestSet(Dataset):
    def __init__(self, args):
        all_list = []
        for i in range(1,10):
            input_list = sorted(glob.glob(os.path.join(args.testdata_root+'/*_{}0{}.h5'.format(args.val_modal,i))))
            if len(input_list)>0:
                print('found {}, repetition #{}: {} subjects'.format(args.val_modal, i, len(input_list)))
                all_list.append(input_list)
        input_list1 = all_list[0]
        
        self.all = all_list
        #probe the image dimensions
        example_image = read_h5(input_list1[0])
        print(f'the image dimension is {example_image.shape}')
        self.images = np.zeros([len(input_list1),len(all_list), 
                                example_image.shape[-3], 
                                example_image.shape[-2],
                                example_image.shape[-1]])
        print('TestSet loading...')
        for i in range(len(self.all)):
            for j,path in enumerate(all_list[i]):
                self.images[j][i] = read_h5(path)
        self.labels = np.mean(self.images,axis=1)
        print('Finished loading')
        
        self.images = self.images.transpose(0,2,1,3,4).reshape(-1,len(all_list), example_image.shape[-2], example_image.shape[-1])
        self.labels = self.labels.reshape(-1,1,example_image.shape[-2], example_image.shape[-1])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        rep_choice = np.random.choice([len(images)//2],1)
        images = images[rep_choice]
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
        noise = np.random.normal(0, np.random.randint(8,15), (h, w, 1))
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

class PNGDataset(Dataset):
    def __init__(self, folder_path, split_ratio=0.95, train=True, image_size=256, noise_std_range=(0.02, 0.06), seed=42):
        # Get all PNG files from the folder
        self.all_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        
        # Split the files into training and testing sets
        random.seed(seed)  # Ensure reproducibility of split
        random.shuffle(self.all_files)
        split_index = int(len(self.all_files) * split_ratio)
        
        if train:
            self.files = self.all_files[:split_index]
        else:
            self.files = self.all_files[split_index:]
        
        # Save the noise settings
        self.noise_std_range = noise_std_range
        self.fixed_noise = not train  # Fix noise for the test set
        self.seed = seed
        
        # Transformation pipeline
        if train:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.RandomChoice([
                    transforms.Resize(int(image_size*0.9)),
                    transforms.Resize(int(image_size)),  # Resize the smaller edge to 256 while keeping the aspect ratio
                ]),
                transforms.CenterCrop(image_size),  # Center crop to a square of size `image_size`
                transforms.RandomChoice([
                    transforms.RandomRotation(degrees=(0, 0)),  # No rotation
                    transforms.RandomRotation(degrees=(90, 90)),  # 90 degrees rotation
                    transforms.RandomRotation(degrees=(180, 180)),  # 180 degrees rotation
                    transforms.RandomRotation(degrees=(270, 270))  # 270 degrees rotation
                ]),
                transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.Resize(image_size),  # Resize the smaller edge to 256 while keeping the aspect ratio
                transforms.CenterCrop(image_size),  # Center crop to a square of size `image_size`
                transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
            ])

    def add_rician_noise(self, image, std):
        """
        Adds Rician noise to the image.
        
        Args:
            image (Tensor): The input image.
            std (float): The standard deviation of the Gaussian noise to be added.
        
        Returns:
            Tensor: The noisy image with Rician noise.
        """
        noise_real = torch.randn_like(image) * std
        noise_imag = torch.randn_like(image) * std
        noisy_image = torch.sqrt((image + noise_real) ** 2 + noise_imag ** 2)
        return torch.clamp(noisy_image, 0, 1)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.files[idx]
        image = Image.open(img_path)
        
        # Apply the transformation pipeline
        image = self.transform(image)
        
        # If fixed_noise is True, set a fixed seed to ensure consistent noise
        if self.fixed_noise:
            torch.manual_seed(self.seed + idx)  # Use idx to ensure different noise per image
        
        # Randomly select a noise standard deviation within the specified range
        noise_std = random.uniform(*self.noise_std_range)
        
        # Add Rician noise to the input image
        noisy_image = self.add_rician_noise(image, noise_std)
        
        # The label remains the original clean image
        label = image
        
        # Return the noisy image and the clean image as the label
        return {'images': noisy_image, 'labels': label}

# Example usage:
# train_dataset = PNGDataset(folder_path='/path/to/train_data', train=True)
# test_dataset = PNGDataset(folder_path='/path/to/test_data', train=False)
