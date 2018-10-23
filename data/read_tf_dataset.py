import tensorflow as tf

MAX_STROKE_COUNT = 256


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
    # The type is now uint8 but we need it to be float.
    # Get the label associated with the image.
    label = parsed_example['targets']
    # The image and label are now correct TensorFlow types.
    return inputs, input_shape, label


def get_iterator(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator