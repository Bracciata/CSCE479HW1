from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
model.add(tf.keras.layers.Dense(90, tf.nn.relu))
model.add(tf.keras.layers.Dense(50, tf.nn.relu))
model.add(tf.keras.layers.Dense(10, tf.nn.softmax))

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
