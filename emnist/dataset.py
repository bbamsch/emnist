"""
Based on official tensorflow MNIST model loader found at
https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py
"""
import os
import tensorflow as tf
import numpy as np


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    data_type = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=data_type)[0]


def read_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as file:
        magic = read32(file)
        num_images = read32(file)
        rows = read32(file)
        cols = read32(file)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                           file.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (file.name, rows, cols))
    return num_images, rows, cols


def read_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as file:
        magic = read32(file)
        num_labels = read32(file)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (magic, file.name))
    return num_labels


def dataset(directory, images_file, labels_file):
    """tf.data.Dataset object for EMNIST images and labels file pair."""
    images_file = os.path.join(directory, images_file)
    labels_file = os.path.join(directory, labels_file)

    if not tf.gfile.Exists(images_file):
        raise FileNotFoundError('Images File was not found')
    if not tf.gfile.Exists(labels_file):
        raise FileNotFoundError('Labels File was not found')

    num_images, _, _ = read_image_file_header(images_file)
    num_labels = read_labels_file_header(labels_file)
    if num_images != num_labels:
        raise RuntimeError('Mismatched number of images and labels')

    def decode_image(image):
        """Normalize from [0, 255] to [0.0, 1.0]"""
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        """Transform label to one hot encoding"""
        label = tf.decode_raw(label, tf.uint8)
        label = tf.one_hot(label, 10)
        label = tf.reshape(label, [10])
        return label

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
