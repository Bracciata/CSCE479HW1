from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util_CIFAR import DisplayUtils
DATA_DIR = './tensorflow-datasets/'
# Gather the data
train = tfds.load(
    'cifar100', split='train[:90%]', data_dir=DATA_DIR, shuffle_files=True)

# After the training loop, run another loop over this data without the gradient updates to calculate accuracy
validation = tfds.load('cifar100', split='train[-10%:]', data_dir=DATA_DIR)
# Looping through the iterator, each batch is a dict
for batch in train:
    print("data shape:", batch['image'].shape)
    print("label:", batch['label'])
    break
displayer = DisplayUtils()
displayer.visualize_batch(batch)
