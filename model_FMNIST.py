class Dense(tf.Module):
    def __init__(self, output_size, activation=tf.nn.relu):
        
        super(Dense, self).__init__(name=name)
        self.output_size = output_size
        self.activation = activation
        self.is_built = False
        
    def _build(self, x):
        data_size = x.shape[-1]
        self.W = tf.Variable(tf.random.normal([data_size, self.output_size]), name='weights')
        self.b = tf.Variable(tf.random.normal([self.output_size]), name='bias')
        self.is_built = True

    def __call__(self, x):
        if not self.is_built:
            self._build(x)
        return self.activation(tf.matmul(x, self.W) + self.b)
    
    
    
    
L2_COEFF = 0.1 # Controls how strongly to use regularization

class L2DenseNetwork(tf.Module):
    def __init__(self, name=None):
        super(L2DenseNetwork, self).__init__(name=name) # remember this call to initialize the superclass
        self.dense_layer1 = tf.keras.layers.Dense(200, activation=tf.nn.relu)
        self.dense_layer2 = tf.keras.layers.Dense(10)
        
    def l2_loss(self):
        # Make sure the network has been called at least once to initialize the dense layer kernels
        return tf.nn.l2_loss(self.dense_layer1.kernel) + tf.nn.l2_loss(self.dense_layer2.kernel)

    @tf.function
    def __call__(self, x):
        embed = self.dense_layer1(x)
        output = self.dense_layer2(embed)
        return output