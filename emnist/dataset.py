import os
import tensorflow as tf
import numpy as np


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           f.name))


def dataset(directory, images_file, labels_file):
    images_file = os.path.join(directory, images_file)
    labels_file = os.path.join(directory, labels_file)

    if not tf.gfile.Exists(images_file):
        raise FileNotFoundError('Images File was not found')
    if not tf.gfile.Exists(labels_file):
        raise FileNotFoundError('Labels File was not found')

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(
        images_file, 28 * 28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
        labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train(directory, splitname):
    """tf.data.Dataset object for EMNIST training data."""
    images_file = '{}-train-images-idx3-ubyte'.format(splitname)
    labels_file = '{}-train-labels-idx1-ubyte'.format(splitname)
    return dataset(directory, images_file, labels_file)


def test(directory, splitname):
    """tf.data.Dataset object for EMNIST test data."""
    images_file = '{}-test-images-idx3-ubyte'.format(splitname)
    labels_file = '{}-test-labels-idx1-ubyte'.format(splitname)
    return dataset(directory, images_file, labels_file)
