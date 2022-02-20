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
