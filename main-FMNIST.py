from __future__ import print_function

import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots
from tqdm import tqdm              # to track progress of loops
import model_FMNIST
DATA_DIR = './tensorflow-datasets/'

ds = tfds.load('fashion_mnist', data_dir=DATA_DIR, shuffle_files=True) # this loads a dict with the datasets

# We can create an iterator from each dataset
# This one iterates through the train data, shuffling and minibatching by 32
train_ds = ds['train'].shuffle(1024).batch(32)

test = tfds.load('fashion_mnist', split='train[:90%]', data_dir=DATA_DIR, shuffle_files=True)

# After the training loop, run another loop over this data without the gradient updates to calculate accuracy
validation = tfds.load('fashion_mnist', split='train[-10%:]', data_dir=DATA_DIR)

#train_test = test['train'].shuffle(1024).batch(32)

# Looping through the iterator, each batch is a dict
for batch in train_ds:
    # The first dimension in the shape is the batch dimension
    # The second and third dimensions are height and width
    # Being greyscale means that the image has one channel, the last dimension in the shape
    print("data shape:", batch['image'].shape)
    print("label:", batch['label'])
    break

# visualize some of the data
idx = np.random.randint(batch['image'].shape[0])
print("idx is: ", idx)
print("An image looks like this:")
imgplot = plt.imshow(batch['image'][idx])
print("It's colored because matplotlib wants to provide more contrast than just greys")


    
# Defining, creating and calling the network repeatedly will trigger a WARNING about re-tracing the function
# So we'll check to see if the variable exists already
if 'l2_dense_net' not in locals():
    l2_dense_net = L2DenseNetwork()
l2_dense_net(tf.ones([1,100]))

l2_loss = l2_dense_net.l2_loss()                     # calculate l2 regularization loss
cross_entropy_loss = 0.                              # calculate the classification loss
total_loss = cross_entropy_loss + L2_COEFF * l2_loss # and add to the total loss, then calculate gradients  

print("Total loss after regularization is: ", total_loss)

# %%
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(200, tf.nn.relu))
model.add(tf.keras.layers.Dense(10))

optimizer = tf.keras.optimizers.Adam()

loss_values = []
accuracy_values = []
# Loop through one epoch of data
for epoch in range(1):
    for batch in tqdm(train_ds):
        with tf.GradientTape() as tape:
            # run network
            x = tf.reshape(tf.cast(batch['image'], tf.float32), [-1, 784])
            labels = batch['label']
            logits = model(x)

            # calculate loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) 
            
        loss_values.append(loss)
    
        # gradient update to check where the loss is minimum
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        # calculate accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        accuracy_values.append(accuracy)

print(model.summary())
    
# accuracy
print("Accuracy:", np.mean(accuracy_values))
# plot per-datum loss
loss_values = np.concatenate(loss_values)
plt.hist(loss_values, density=True, bins=30)
plt.show()

# %%



