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
        pool_1 = tf.keras.layers.MaxPool2D(padding='same')
        hidden_3 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_3')
        hidden_4 = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='hidden_4')
        pool_2 = tf.keras.layers.MaxPool2D(padding='same')
        flatten = tf.keras.layers.Flatten()
        output = tf.keras.layers.Dense(10)
        self.conv_classifier = tf.keras.Sequential(
            [hidden_1, hidden_2, pool_1, hidden_3, hidden_4, pool_2, flatten, output])

    def train(self, ds):
        # Run some data through the network to initialize it
        for batch in ds:
            # data is uint8 by default, so we have to cast it
            self.conv_classifier(tf.cast(batch['image'], tf.float32))
            break
