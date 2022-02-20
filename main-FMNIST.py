from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


DATA_DIR = './tensorflow-datasets/'


ds = tfds.load('fashion_mnist', data_dir=DATA_DIR, shuffle_files=True)
train_ds = ds['train'].shuffle(1024).batch(16)


train = tfds.load('fashion_mnist', split='train[:90%]', data_dir=DATA_DIR)
validation = tfds.load(
    'fashion_mnist', split='train[-10%:]', data_dir=DATA_DIR)

val_ds = ds['train'].shuffle(1024).batch(16)

class_names = ["T-shirts", "Trousers", "Pullovers",
               "Dress", "Coat", "Sandal", "Shirt", "Bag", "Ankle boot"]


for batch in train_ds:
    print("data shape:", batch['image'].shape)
    print("data shape:", batch['image'].dtype)
    labl = batch['label']
    print("label:", labl)
    break


test = batch['image'][5000:]
# visualize some of the data

idx = np.random.randint(batch['label'].shape[0])
idx1 = np.random.randint(batch['image'].shape[0])
idx2 = np.random.randint(batch['image'].shape)
idx3 = np.random.randint(batch['image'].shape[2])


print("idx is: ", idx)
#print("idx2 is: ", idx2)
#print("idx3 is: ", idx3)

#print("batch image: ", batch['image'][idx])

#print("batch image is: ", batch['image'][idx])

print("An image looks like this:")
imgplot = plt.imshow(batch['image'][idx])
#imgplot2 = plt.imshow(batch['image'][idx3])


print("It's colored because matplotlib wants to provide more contrast than just greys")

print("label number is: ", labl[idx])


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

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
model.add(tf.keras.layers.Dense(90, tf.nn.relu))
model.add(tf.keras.layers.Dense(50, tf.nn.relu))
model.add(tf.keras.layers.Dense(10, tf.nn.softmax))

#model.compile(optimizer="adam", loss="sparse_softmax_crossentropy", metrics=["accuracy"])
#history = model.fit(train,epochs=30,validation_data=(validation))

optimizer = tf.keras.optimizers.Adam()

loss_values = []
accuracy_values = []


for epoch in range(1):
    for batch in tqdm(train_ds):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            x = x / 255.0
            labels = batch['label']
            logits = model(x)

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
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, labels), tf.float32))
        accuracy_values.append(accuracy)


for epoch in range(1):
    for batch in tqdm(val_ds):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            x = x / 255.0
            labels = batch['label']
            logits = model(x)

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


output_layer = model.layers[3]
weights = output_layer.get_weights()
#print("weights: ", weights)

# print(x)
print(model.summary())

# model.compile(metrics=["accuracy"])

print("Confusion matrix: ")
print(tf.math.confusion_matrix(labels, predictions))

# accuracy
print("Accuracy:", np.mean(accuracy_values))
# plot per-datum loss
loss_values = np.concatenate(loss_values)
plt.hist(loss_values, density=True, bins=30)
plt.show()

#print("it's a ", class_names[batch['label'].shape[0]]);
