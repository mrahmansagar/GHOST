# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:20:57 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

Training a pix2pix with data from rat lungs 
"""

import os 
os.sys.path.insert(0, 'E:\\dev\\packages')
# from proUtils import utils
from glob import glob
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from sklearn.feature_extraction.image import extract_patches_2d

from GANs import utils as u
from GANs.pix2pix import models

ct_dir = 'D:\\Rat_Elastic_Reg\\CT_cropped\\'
picro_dir = 'D:\\Rat_Elastic_Reg\\Picro_cropped\\'


ct_slides = glob(ct_dir + '*')
picro_slides = glob(picro_dir + '*')


# visualization 
plt.figure()   
a = Image.open(ct_slides[0])
a = np.array(a)
plt.imshow(a, cmap='gray')
plt.show()

plt.figure()
h = Image.open(picro_slides[0])
h = np.array(h)
plt.imshow(h,)
plt.show()



def number_of_patches(img_shape, patch_size):
    row = int(img_shape[0]/patch_size[0])
    column = int(img_shape[1]/patch_size[1])
    
    return row*column
    
patch_size = (256, 256)

# creating training data
# train data Aperion
list_of_ct_patches = []
list_of_histo_patches = []
for ct, histo in tqdm(zip(ct_slides, picro_slides)):
    ct_img = Image.open(ct)
    ct_img = np.array(ct_img)
    
    histo_img = Image.open(histo)
    histo_img = np.array(histo_img)
    
    nb_of_patches = number_of_patches(ct_img.shape, patch_size=patch_size)
    ct_patches = extract_patches_2d(ct_img, patch_size=patch_size, max_patches=nb_of_patches*5, random_state=41)
    ct_patches = np.expand_dims(ct_patches, axis=-1)
    histo_patches = extract_patches_2d(histo_img, patch_size=patch_size, max_patches=nb_of_patches*5, random_state=41)
    list_of_ct_patches.append(ct_patches)
    list_of_histo_patches.append(histo_patches)
    
trainCT = np.concatenate(list_of_ct_patches, axis=0)
trainHisto = np.concatenate(list_of_histo_patches, axis=0)


idxA = np.random.randint(0, len(ct_patches), 5)

plt.figure(figsize=(10,5))
for i, idx in enumerate(idxA):
    plt.subplot(1, 5, 1 + i)
    plt.axis('off')
    plt.imshow(trainCT[idx], cmap='gray')
plt.show()
plt.figure(figsize=(10,5))
for i, idx in enumerate(idxA):
    plt.subplot(1, 5, 1 + i)
    plt.axis('off')
    plt.imshow(trainHisto[idx])
plt.show()


src_CT = u.scale_data(trainCT)
tar_Histo = u.scale_data(trainHisto)


src_shape = src_CT.shape[1:]
tar_shape = tar_Histo.shape[1:]

tar_channel = tar_Histo.shape[-1]

dis = models.build_discriminator(src_shape=src_shape, tar_shape=tar_shape)
gen = models.build_generator(input_shape=src_shape, output_channel=tar_channel)


p2p_model = models.build_pix2pix(gen, dis, input_shape=src_shape)

# train 
models.train_pix2pix(gen, dis, p2p_model, src_CT, tar_Histo, epochs=100, summary_interval=2, name='CT2Picro')



