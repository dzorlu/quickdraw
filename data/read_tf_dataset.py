# coding=utf-8
import sys
import os
import tensorflow as tf
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from data import create_tf_dataset


def get_filenames(mode):
    return "quickdraw-{}*".format(mode)


def parse_fn(mode):
    def parse(serialized):
        if mode == 'test':
            features = \
                {
                    'inputs': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                }
        else:
            features = \
                {
                    'inputs': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
                    'targets': tf.FixedLenFeature([1], tf.int64)
                }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized, features=features)
        # Get the image as raw bytes.
        inputs = parsed_example['inputs']
        # pad sequences leads to a fixed size. here we reshape back.
        inputs = tf.reshape(inputs, (create_tf_dataset.MAX_STROKE_COUNT, 3))
        # Get the label associated with the image.
        if mode != 'test':
            targets = parsed_example['targets']
            targets = tf.one_hot(targets, create_tf_dataset.NB_CLASSES)
            targets = tf.squeeze(targets)
            return inputs, targets
        return inputs
    return parse


def get_iterator(data_dir, mode, batch_size=32):
    filenames = get_filenames(mode)
    filepaths = glob.glob(os.path.join(data_dir, filenames))
    dataset = tf.data.TFRecordDataset(filepaths)
    parse = parse_fn(mode)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset.make_one_shot_iterator()


class Iter(object):
    def __init__(self, data_dir, mode, batch_size):
        self.data_dir = data_dir
        self.mode = mode
        self.batch_size = batch_size
        self.input_fn = get_iterator(self.data_dir, mode, self.batch_size)
    def __iter__(self):
        return self
    def __next__(self):
        try:
            self.input_fn.get_next()
        except IndexError:
            raise StopIteration



