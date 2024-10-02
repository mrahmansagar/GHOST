# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:04:39 2024

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

Training a cycleGAN using the dataset used in the StainGAN paper. 
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
from GANs.cycleGAN import models

# directory where the training slides are
scan_aperio_dir = 'D:\\StainGAN\\mitos_atypia_2014_training_aperio\\'
scan_hama_dir = 'D:\\StainGAN\\mitos_atypia_2014_training_hamamatsu\\'


aperio_scans = os.listdir(scan_aperio_dir)
hama_scans = os.listdir(scan_hama_dir)

aperio_scans_path = []
hama_scans_path = []

# full paths of the slides 
for scan in aperio_scans:
    aperio_scans_path += (glob(os.path.join(scan_aperio_dir, scan) + "\\frames\\x20\\*" ))
    
# full paths of the slides     
for scan in hama_scans:
    hama_scans_path += (glob(os.path.join(scan_hama_dir, scan) + "\\frames\\x20\\*" ))


# visualization 
plt.figure()   
a = Image.open(aperio_scans_path[0])
a = np.array(a)
plt.imshow(a,)
plt.show()

plt.figure()
h = Image.open(hama_scans_path[0])
h = np.array(h)
plt.imshow(h,)
plt.show()



# creating training data
# train data Aperion
list_of_aperio_patches = []
for ap in tqdm(aperio_scans_path):
    img = Image.open(ap)
    img = np.array(img)
    patches = extract_patches_2d(img, patch_size=(256, 256), max_patches=33, random_state=41)
    list_of_aperio_patches.append(patches)
    
trainAperio = np.concatenate(list_of_aperio_patches, axis=0)


list_of_hama_patches = []
for ha in tqdm(hama_scans_path):
    img = Image.open(ha)
    img = img.resize((1539, 1376), Image.LANCZOS)
    img = np.array(img)
    patches = extract_patches_2d(img, patch_size=(256, 256), max_patches=33, random_state=41)
    list_of_hama_patches.append(patches)
    
    
trainHama = np.concatenate(list_of_hama_patches, axis=0)

idxA = np.random.randint(0, len(trainAperio), 5)
idxB = np.random.randint(0, len(trainHama), 5)

plt.figure(figsize=(10,5))
for i, idx in enumerate(idxA):
    plt.subplot(1, 5, 1 + i)
    plt.axis('off')
    plt.imshow(trainAperio[idx].astype('uint8'))
plt.show()
plt.figure(figsize=(10,5))
for i, idx in enumerate(idxB):
    plt.subplot(1, 5, 1 + i)
    plt.axis('off')
    plt.imshow(trainHama[idx].astype('uint8'))
plt.show()



#=============================================================================#

# data preprocessing 
trainAperio = u.scale_data(trainAperio[0:10])
trainHama = u.scale_data(trainHama[0:10])

# input shape
image_shape = trainAperio.shape[1:]

# generator model
genAperio2Hama = models.build_generator(input_shape=image_shape)
genHama2Aperio = models.build_generator(input_shape=image_shape)

# discriminator model
disAperio = models.build_discriminator(input_shape=image_shape)
dis_Hama = models.build_discriminator(input_shape=image_shape)

# cycleGAN
cycleGAN_Aperio2Hama = models.build_cycleGAN(genAperio2Hama, dis_Hama, genHama2Aperio, input_shape=image_shape)
cycleGAN_Hama2Aperio = models.build_cycleGAN(genHama2Aperio, disAperio, genAperio2Hama, input_shape=image_shape)


#Training 
models.train_cycleGAN(disAperio, dis_Hama, genAperio2Hama, genHama2Aperio, 
                      cycleGAN_Aperio2Hama, cycleGAN_Hama2Aperio, trainAperio, trainHama,
                      batch_size=1, epochs=2, summary_interval=1, nameA2B='GenAperio2Hama', nameB2A='GenHama2Aperio')

