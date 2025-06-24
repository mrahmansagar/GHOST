# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 15:42:24 2025

@author: mrahman Sagar

Colon CT to picro
"""


import os
os.sys.path.insert(0, 'E:\\dev\\packages')

from tqdm import tqdm 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 


from PIL import Image 

from GANs import utils as u
from GANs.pix2pix import models


root_dir = "E:\\Data\\ghost\\"

# Read the CSV file with cross entropy values between ct and histo pairs
df = pd.read_csv('pairs_with_cross_entropy.csv')

max_cross_entropy = 0.02

filtered_df = df[df['CE'] <= max_cross_entropy]

# reading CT and histo patches
ct_patches = []
histo_patches = []
ct_mins = []
ct_maxs = []
histo_mins = []
histo_maxs = []

for _, row in tqdm(filtered_df.iterrows()):
    ct_img = Image.open(row[0])
    ct_img = np.array(ct_img)
    ct_mins.append(np.min(ct_img.flatten()))
    ct_maxs.append(np.max(ct_img.flatten()))
    ct_patches.append(ct_img)
    
    histo_img = Image.open(row[1])
    histo_img = histo_img.convert('RGB')
    histo_img = np.array(histo_img)
    histo_mins.append(np.min(histo_img.flatten()))
    histo_maxs.append(np.max(histo_img.flatten()))
    histo_patches.append(histo_img)
    

ct_min_value = np.min(ct_mins)
ct_max_value = np.max(ct_maxs)

ct_patches = np.array(ct_patches)
ct_patches = np.expand_dims(ct_patches, axis=-1)  


histo_min_value = np.min(histo_mins)
histo_max_value = np.max(histo_maxs)

histo_patches = np.array(histo_patches)

  
idxA = np.random.randint(0, len(ct_patches), 5)

plt.figure(figsize=(10,5))
for i, idx in enumerate(idxA):
    plt.subplot(1, 5, 1 + i)
    plt.axis('off')
    plt.imshow(ct_patches[idx], cmap='gray')
plt.show()
plt.figure(figsize=(10,5))
for i, idx in enumerate(idxA):
    plt.subplot(1, 5, 1 + i)
    plt.axis('off')
    plt.imshow(histo_patches[idx])
plt.show()


src_CT = u.scale_data(ct_patches[0:400], ct_min_value, ct_max_value)
tar_Histo = u.scale_data(histo_patches[0:400], histo_min_value, histo_max_value)



src_shape = src_CT.shape[1:]
tar_shape = tar_Histo.shape[1:]

tar_channel = tar_Histo.shape[-1]

dis = models.build_discriminator(src_shape=src_shape, tar_shape=tar_shape)
gen = models.build_generator(input_shape=src_shape, output_channel=tar_channel)


p2p_model = models.build_pix2pix(gen, dis)

# train 
models.train_pix2pix(gen, dis, p2p_model, src_CT, tar_Histo, epochs=500, summary_interval=10, name='colon_CT2Histo_Norm')


# model evaluation 
from keras.models import load_model

model = load_model("E:\\projects\\GHOST\\colon_CT2Histo_202506231604\\model_after_457000.h5")

test_df = df[df['CE'] >= max_cross_entropy]

# reading CT and histo patches
ct_patches = []
histo_patches = []

for _, row in tqdm(test_df.iterrows()):
    ct_img = Image.open(row[0])
    ct_img = np.array(ct_img)
    ct_patches.append(ct_img)
    
    histo_img = Image.open(row[1])
    histo_img = histo_img.convert('RGB')
    histo_img = np.array(histo_img)
    histo_patches.append(histo_img)
    
ct_patches = np.array(ct_patches)
ct_patches = np.expand_dims(ct_patches, axis=-1)  

histo_patches = np.array(histo_patches)


def plot_src_gen_tar(src, tar, gen_model, sample_size=5):
    idx = np.random.randint(0, len(src), sample_size)
    
    sel_src = src[idx]
    scaled_src = (sel_src - 127.5) / 127.5
    sel_tar = tar[idx]
    gen = gen_model.predict(scaled_src)
    gen = (gen + 1) / 2.0
    
    
    fig, axes = plt.subplots(3, sample_size, figsize=(10, 7))
    
    for i in range(sample_size):
        axes[0, i].imshow(sel_src[i].astype('uint8'), cmap='gray')
        axes[0, i].set_title('Source Image')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(gen[i])
        axes[1, i].set_title('Generated Image')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(sel_tar[i].astype('uint8'))
        axes[2, i].set_title('Target Image')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()