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
