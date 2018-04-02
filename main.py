"""
Main entry point for machine learning project.
"""
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import emnist

FLAGS = flags.FLAGS

flags.DEFINE_string(
    name='emnist_data',
    default='./data/emnist',
    help='Path to EMNIST data')
flags.DEFINE_float(
    name='learning_rate',
    default='0.05',
    help='Model Learning Rate')


def main(argv):
    """Basic Gradient Descent Learning Model."""
    del argv  # Unused

    # Load Datasets
    training_data = emnist.train(FLAGS.emnist_data, 'emnist-digits')
    testing_data = emnist.test(FLAGS.emnist_data, 'emnist-digits')

    # Initialize Function
    x = tf.placeholder(tf.float32, [None, 28*28])
    W = tf.Variable(tf.zeros([28*28, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Initialize Labels
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Initialize Error Measure
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        batched_dataset = training_data.batch(10000)
        batch_iterator = batched_dataset.make_initializable_iterator()
        features, labels = batch_iterator.get_next()
        for _ in range(10):
            try:
                sess.run(batch_iterator.initializer)
                while True:
                    sess.run(train_step, feed_dict={x: features.eval(), y_: labels.eval()})
            except tf.errors.OutOfRangeError:
                # Done Training
                pass

        logging.info(W.eval())

        batched_dataset = testing_data.batch(1000)
        batch_iterator = batched_dataset.make_one_shot_iterator()
        features, labels = batch_iterator.get_next()
        predictions = sess.run(y, feed_dict={x: features.eval()})
        logging.info(predictions)


if __name__ == '__main__':
    app.run(main)
