# Softmax example in TF using the classical Iris dataset
# Download iris.data from https://archive.ics.uci.edu/ml/datasets/Iris

import tensorflow as tf
from sklearn.datasets import *
from sklearn.model_selection import train_test_split

# this time weights form a matrix, not a column vector, one "weight vector" per class.
W = tf.Variable(tf.zeros([64, 10]), name="weights")
# so do the biases, one per class.
b = tf.Variable(tf.zeros([10], name="bias"))


def combine_inputs(X):
    return tf.matmul(X, W) + b


def inference(X):
    return tf.nn.softmax(combine_inputs(X))


def loss(X, Y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


def input():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
    return tf.to_float(X_train), tf.to_int32(y_train), tf.to_float(X_test), tf.to_int32(y_test)


def train(target_loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(target_loss)


def evaluate(tf_session, X, Y):
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)

    # Launch the graph in a session, setup boilerplate
    return tf_session.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32)))


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    X, Y, x, y = input()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    writer = tf.summary.FileWriter('./softmax_digit_graph', sess.graph)
    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented through training steps
        if step % 10 == 0:
            print("iteration %d loss: %.5f, train accuracy: %.5f, test accuracy: %.5f"
                  % (step, sess.run(total_loss), evaluate(sess, X, Y), evaluate(sess, x, y)))

    coord.request_stop()
    coord.join(threads)
    writer.close()
    sess.close()
