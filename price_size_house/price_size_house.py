import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from createData import Data
from prepareModel import model

# Prepare data
data = Data()

# Prepare model
model = model(data.n_train)

# Initialize environment
init = tf.global_variables_initializer()

# Configure for training
display_every = 2
num_training_iter = 50

# Train with a session
with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_training_iter):

        # Fit data
        for (x, y) in zip(data.train_house_size, data.train_house_price):
            sess.run(model.tf_optimizer, feed_dict={model.tf_house_size: x,
                                                 model.tf_house_price: y})

        # Display
        if (iteration + 1) % display_every == 0:
            est_cost = sess.run(model.tf_cost,
                                feed_dict={model.tf_house_size: data.train_house_size,
                                           model.tf_house_price: data.train_house_price})
            print("iter # ", iteration + 1, "cost", est_cost, "interpret",
                  sess.run(model.tf_interpret), "bias", sess.run(model.tf_bias))

    print("Finished!!!")
    training_cost = sess.run(model.tf_cost,
                             feed_dict={model.tf_house_size: data.train_house_size,
                                        model.tf_house_price: data.train_house_price})
    print("cost", training_cost, "interpret", sess.run(model.tf_interpret),
          "bias", sess.run(model.tf_bias))

    # Plot data
    train_house_size_on_graph = data.train_house_size*data.train_house_size.std()
                                + data.train_house_size.mean()
    train_house_price_on_graph = data.train_house_price*data.train_house_price.std()
                                 + data.train_house_price.mean()
    test_house_size_on_graph = data.test_house_size*data.test_house_size.std()
                               + data.test_house_size.mean()
    test_house_price_on_graph = data.test_house_price*data.test_house_price.std()
                                + data.test_house_price.mean()
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("Price")
    plt.xlabel("Size (sq.ft)")
    plt.plot(test_house_size_on_graph, test_house_price_on_graph,"ro", label="Training data")
    plt.plot(train_house_size_on_graph, train_house_price_on_graph,"bx", label="Testing data")
    plt.plot(train_house_size_on_graph,
             sess.run(model.tf_interpret)*train_house_size_on_graph
             + sess.run(model.tf_bias)*data.train_house_price.std() + data.train_house_price.mean(),
             label="Linear Regression")
    plt.show()
