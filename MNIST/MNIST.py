import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# Pull input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

# Setup placeholders
images = tf.placeholder(tf.float32, shape=[None, 784])
prob = tf.placeholder(tf.float32, shape=[None, 10])

# Setup variables
weights = tf.Variable(tf.zeros([784, 10]))
thresholds = tf.Variable(tf.zeros([10]))

# Inference function
inference_y = tf.matmul(images, weights) + thresholds

# Activate function
estimate_prob = tf.nn.softmax(inference_y)

# Loss measurements
cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=prob, logits=estimate_prob))

# Setup optimizer
learning_rate = 0.5
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Setup environment
init = tf.global_variables_initializer()

# Configure training
n_train_iter = 1000

# Start training
with tf.Session() as sess:

    sess.run(init)

    # Train data as batch training
    for iteration in range(n_train_iter):
        # Get 100 random images
        batch_img, batch_prob = mnist.train.next_batch(100)
        # Fit batch data to model
        sess.run(training_step, feed_dict={images: batch_img, prob: batch_prob})

    # Validate
    isCorrect = tf.equal(tf.argmax(estimate_prob, 1), tf.argmax(prob, 1))
    accuracy_score = tf.reduce_mean(tf.cast(isCorrect, tf.float32))
    test_accuracy = sess.run(accuracy_score,
                        feed_dict={images: mnist.test.images, prob: mnist.test.labels})
    print(test_accuracy * 100)
