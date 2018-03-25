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


def main(argv):
    del argv  # Unused

    logging.info('Start')
    sess = tf.InteractiveSession()
    training_data = emnist.train(FLAGS.emnist_data, 'emnist-byclass')


if __name__ == '__main__':
    app.run(main)
