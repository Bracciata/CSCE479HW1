from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds  # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
from util_CIFAR import EarlyStopping


class Dense(tf.Module):
    def __init__(self, output_size, activation=tf.nn.relu):

        super(Dense, self).__init__(name=name)
        self.output_size = output_size
        self.activation = activation
        self.is_built = False

    def _build(self, x):
        data_size = x.shape[-1]
        self.W = tf.Variable(tf.random.normal(
            [data_size, self.output_size]), name='weights')
        self.b = tf.Variable(tf.random.normal([self.output_size]), name='bias')
        self.is_built = True

    def __call__(self, x):
        if not self.is_built:
            self._build(x)
        return self.activation(tf.matmul(x, self.W) + self.b)


L2_COEFF = 0.1


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


if 'l2_dense_net' not in locals():
    l2_dense_net = L2DenseNetwork()
l2_dense_net(tf.ones([1, 100]))


class ModelOne:
    def __init__(self):
        # Adding Regularlizer based on official Tensorflow Docs.
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
        self.model.add(tf.keras.layers.Dense(300, tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(99, tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(10, tf.nn.softmax))

    def initialize(self, train_ds):
        class_names = ["T-shirts", "Trousers", "Pullovers",
                       "Dress", "Coat", "Sandal", "Shirt", "Bag", "Ankle boot"]

        for batch in train_ds:
            print("data shape:", batch['image'].shape)
            print("data shape:", batch['image'].dtype)
            labl = batch['label']
            print("label:", labl)
            break

    def train(self, train_ds, val_ds):
        optimizer = tf.keras.optimizers.Adam()

        loss_values = []
        accuracy_values = []

        for epoch in range(1):
            for batch in tqdm(train_ds):
                with tf.GradientTape() as tape:
                    # run network
                    x = tf.cast(batch['image'], tf.float32)
                    x = x / 255.0
                    labels = batch['label']
                    logits = self.model(x)

                    # calculate loss

                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels)
                    # calculate l2 regularization loss
                    l2_loss = l2_dense_net.l2_loss()
                    # calculate the classification loss
                    cross_entropy_loss = loss
                    # and add to the total loss, then calculate gradients
                    total_loss = cross_entropy_loss + L2_COEFF * l2_loss

                loss_values.append(total_loss)

                # gradient update to check where the loss is minimum
                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables))

                # calculate accuracy
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(predictions, labels), tf.float32))
                accuracy_values.append(accuracy)

        for epoch in range(1):
            for batch in tqdm(val_ds):
                with tf.GradientTape() as tape:
                    # run network
                    x = tf.cast(batch['image'], tf.float32)
                    x = x / 255.0
                    labels = batch['label']
                    logits = self.model(x)

                    # calculate loss

                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels)
                    l2_loss = l2_dense_net.l2_loss()
                    cross_entropy_loss = loss
                    total_loss = cross_entropy_loss + L2_COEFF * l2_loss

                loss_values.append(total_loss)

                # calculate accuracy
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(predictions, labels), tf.float32))
                accuracy_values.append(accuracy)

        output_layer = self.model.layers[3]
        weights = output_layer.get_weights()
        #print("weights: ", weights)

        # print(x)
        print(self.model.summary())

        # model.compile(metrics=["accuracy"])

        print("Confusion matrix: ")
        print(tf.math.confusion_matrix(labels, predictions))

        # accuracy
        print("Accuracy:", np.mean(accuracy_values))
        # plot per-datum loss
        loss_values = np.concatenate(loss_values)
        plt.hist(loss_values, density=True, bins=30)
        plt.show()
