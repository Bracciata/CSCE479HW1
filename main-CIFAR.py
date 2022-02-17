from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util_CIFAR import DisplayUtils, DataUtils
from model_CIFAR import ModelOne
DATA_DIR = './tensorflow-datasets/'
# Gather the data
# this loads a dict with the datasets
ds = tfds.load('cifar100', shuffle_files=True, data_dir=DATA_DIR)
train_ds = ds['train'].shuffle(1024).batch(32)
#data_displayer = DataUtils()
#last_batch = data_displayer.describe_data(train_ds)
#displayer = DisplayUtils()
# displayer.visualize_batch(last_batch)
model_one = ModelOne()
model_one.initialize(train_ds)
model_one.train(train_ds)
