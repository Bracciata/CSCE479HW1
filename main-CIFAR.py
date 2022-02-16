from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util_CIFAR import DisplayUtils
DATA_DIR = './tensorflow-datasets/'
# Gather the data
# this loads a dict with the datasets
ds = tfds.load('cifar100', shuffle_files=True, data_dir=DATA_DIR)
train_ds = ds['train'].shuffle(1024).batch(32)
# Looping through the iterator, each batch is a dict
for batch in train_ds:
    print("data shape:", batch['image'].shape)
    print("label:", batch['label'])
    break
displayer = DisplayUtils()
displayer.visualize_batch(batch)
