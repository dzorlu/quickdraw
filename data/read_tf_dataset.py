import tensorflow as tf
import glob
import os

MAX_STROKE_COUNT = 256


def get_filenames(mode):
    return "quickdraw - {} *".format(mode)


def parse(serialized):
    features = \
        {
            'inputs': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'input_shape': tf.FixedLenFeature([1], tf.int64),
            'targets': tf.FixedLenFeature([1], tf.int64)
        }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    # Get the image as raw bytes.
    inputs = parsed_example['inputs']
    # pad sequences leads to a fixed size. here we reshape back.
    inputs = tf.reshape(inputs, (MAX_STROKE_COUNT, 3))
    input_shape = parsed_example['input_shape']
    # Get the label associated with the image.
    label = parsed_example['targets']
    # The image and label are now correct TensorFlow types.
    return inputs, label


def get_iterator(data_dir, mode, batch_size):
    filenames = get_filenames(mode)
    filepaths = glob.glob(os.path.join(data_dir, filenames))
    dataset = tf.data.TFRecordDataset(filepaths)
    dataset = dataset.map(parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator