from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops


class DisplayUtils:
    def visualize_batch(self, batch):
        # visualize some of the data
        idx = np.random.randint(batch['image'].shape[0])
        print("idx is: ", idx)
        imgplot = plt.imshow(batch['image'][idx])
        plt.show()
