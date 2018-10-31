# coding=utf-8
import sys
import os
import tensorflow as tf
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from data import create_tf_dataset

def get_filenames(mode):
    return "quickdraw-{}*".format(mode)


def parse(serialized):
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
    label = parsed_example['targets']
    label = tf.one_hot(label, create_tf_dataset.NB_CLASSES)
    label = tf.squeeze(label)
    # The image and label are now correct TensorFlow types.
    return inputs, label


def get_iterator(data_dir, mode, batch_size):
    filenames = get_filenames(mode)
    filepaths = glob.glob(os.path.join(data_dir, filenames))
    dataset = tf.data.TFRecordDataset(filepaths)
    dataset = dataset.map(parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset.make_one_shot_iterator()

