from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from util_FMNIST import DisplayUtils, DataUtils
from model_FMNIST import ModelOne

DATA_DIR = './tensorflow-datasets/'

train = tfds.load(
    'fashion_mnist', split='train[:90%]', data_dir=DATA_DIR).shuffle(1024).batch(16)
validation = tfds.load(
    'fashion_mnist', split='train[-10%:]', data_dir=DATA_DIR).shuffle(1024).batch(16)

data_displayer = DataUtils()
last_batch = data_displayer.describe_data(train)
displayer = DisplayUtils()
displayer.visualize_batch(last_batch)
model_one = ModelOne()
model_one.initialize(train)
model_one.train(train, validation)
