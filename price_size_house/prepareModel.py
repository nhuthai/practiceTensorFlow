import tensorflow as tf
import numpy as np

class model():
    def __init__(self, n_train):
        # Set up placeholders
        self.tf_house_price = tf.placeholder("float", name="house_price")
        self.tf_house_size = tf.placeholder("float", name="house_size")

        # Set up variables or coefficients
        self.tf_interpret = tf.Variable(np.random.randn(), name="interpret")
        self.tf_bias = tf.Variable(np.random.randn(), name="bias")

        # Set up inference function
        self.tf_price_prediction = tf.add(tf.multiply(self.tf_interpret,
                                                      self.tf_house_size),
                                          self.tf_bias)

        # Set up loss measurement
        self.tf_cost = tf.reduce_sum(tf.pow(self.tf_price_prediction
                                            - self.tf_house_price, 2)) / (2*n_train)

        # Set up optimizer
        learning_rate = 0.1

        self.tf_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.tf_cost)
