from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops


class ModelOne:
    def __init__(self):
        hidden_1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_1')
        hidden_2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_2')
        flatten = tf.keras.layers.Flatten()
        output = tf.keras.layers.Dense(10)
        conv_classifier = tf.keras.Sequential(
            [hidden_1, hidden_2, flatten, output])
