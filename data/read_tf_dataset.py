import tensorflow as tf


def parse(serialized):
    features = \
        {
            'inputs': tf.FixedLenFeature([], tf.string),
            'input_shape': tf.FixedLenFeature([], tf.int64),
            'targets': tf.FixedLenFeature([], tf.int64)
        }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)
    # Get the image as raw bytes.
    inputs = parsed_example['inputs']
    input_shape = parsed_example['input_shape']
    # The type is now uint8 but we need it to be float.
    inputs = tf.decode_raw(inputs, tf.uint8)
    # Get the label associated with the image.
    label = parsed_example['targets']
    # The image and label are now correct TensorFlow types.
    return inputs, input_shape, label


def read_tf_records(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs, input_shape, label = iterator.get_next()
    return inputs, input_shape, label