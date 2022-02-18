from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops


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
        
class EarlyStopping:
    def __init__(self, patience=5, epsilon=1e-4):
        """
        Args:
            patience (int): how many epochs of not improving before stopping training
            epsilon (float): minimum amount of improvement required to reset counter
        """
        self.patience = patience
        self.epsilon = epsilon
        self.best_loss = float('inf')
        self.epochs_waited = 0
    
    def __str__(self):
        return "Early stopping has waited {} epochs out of {} at loss {}".format(self.epochs_waited, self.patience, self.best_loss)
        
    def check(self, loss):
        """
        Call after each epoch to check whether training should halt
        
        Args:
            loss (float): loss value from the most recent epoch of training
            
        Returns:
            True if training should halt, False otherwise
        """
        if loss < (self.best_loss - self.epsilon):
            self.best_loss = loss
            self.epochs_waited = 0
            return False
        else:
            self.epochs_waited += 1
            return self.epochs_waited > self.patience