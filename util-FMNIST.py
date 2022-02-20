from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


class DisplayUtils:
    def visualize_batch(self, batch):
        # visualize some of the data, pick randomly every time this cell is run
        idx = np.random.randint(batch['image'].shape[0])
        print("An image looks like this:")
        imgplot = plt.imshow(batch['image'][idx])
        plt.show()


class DataUtils:
    def describe_data(self, ds):
        # Looping through the iterator, each batch is a dict
        for batch in ds:
            print("data shape:", batch['image'].shape)
            print("label:", batch['label'])
            break
        return batch


class L2DenseNetwork(tf.Module):
    def __init__(self, name=None):
        super(L2DenseNetwork, self).__init__(name=name)
        self.dense_layer1 = tf.keras.layers.Dense(200, activation=tf.nn.relu)
        self.dense_layer2 = tf.keras.layers.Dense(10)
        self.is_built = False

    def _build(self, x):
        self.is_built = True

    def l2_loss(self):
        return tf.nn.l2_loss(self.dense_layer1.kernel) + tf.nn.l2_loss(self.dense_layer2.kernel)

    @tf.function
    def __call__(self, x):
        if not self.is_built:
            self._build(x)
        embed = self.dense_layer1(x)
        output = self.dense_layer2(embed)
        #print("output loss: ", output)
        return output
