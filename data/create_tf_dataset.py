#!/home/deniz/anaconda3/envs/tensorflow_gpuenv/bin/python

import argparse
import os
import glob
import json
import six
import cv2
from sklearn.externals import joblib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensor2tensor.data_generators import generator_utils
from keras.preprocessing import sequence


MAX_STROKE_COUNT = 256
NB_CLASSES = 340
ENCODER_NAME = 'word_encoder.pkl'
TEST_FILE = 'test_simplified.csv'


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if isinstance(v[0], six.integer_types) or isinstance(v[0], np.int64) or isinstance(v[0], np.int32):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      if not six.PY2:  # Convert in python 3.
        v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif isinstance(v[0], bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=features))


def _generate_files(generator, output_filenames,
                   max_cases=None, cycle_every_n=1):
  """Generate cases from a generator and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_filenames: List of output file paths.
    max_cases: maximum number of cases to get from the generator;
      if None (default), we use the generator until StopIteration is raised.
    cycle_every_n: how many cases from the generator to take before
      switching to the next shard; by default set to 1, switch every case.
  """
  tmp_filenames = [fname + ".incomplete" for fname in output_filenames]
  num_shards = len(output_filenames)
  writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filenames]
  counter, shard = 0, 0
  for case in generator:
    if case is None:
      continue
    if counter % 100000 == 0:
      tf.logging.info("Generating case %d." % counter)
    counter += 1
    if max_cases and counter > max_cases:
      break
    example = to_example(case)
    writers[shard].write(example.SerializeToString())
    if counter % cycle_every_n == 0:
      shard = (shard + 1) % num_shards

  for writer in writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filenames, output_filenames):
    tf.gfile.Rename(tmp_name, final_name)

  tf.logging.info("Generated %s Examples", counter)


def generate_samples(task, file_path, is_train=True):
    """
    load and process the csv files - each csv file contains examples belonging to a class
    """
    df = pd.read_csv(file_path)
    df['drawing'] = [json.loads(draw) for draw in df['drawing'].values]
    _map_fn = _stack_it if task == 'strokes' else _draw_it
    x = df['drawing'].map(_map_fn).values
    if is_train:
        y = df['word'].values

        def _generate_samples():
            for _x, _y in zip(x, y):
                yield {'targets': [int(_y)],
                       'inputs': _x.reshape(-1)}
    else:
        def _generate_samples():
            for _x in x:
                yield {'inputs': _x.reshape(-1)}
    return _generate_samples()


def _draw_it(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((size, size), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    return img


def _stack_it(raw_strokes):
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    # unwrap the list
    in_strokes = [(xi, yi, i) for i, (x, y) in enumerate(raw_strokes) for xi, yi in zip(x, y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new stroke
    c_strokes[:, 2] = [1] + np.diff(c_strokes[:,2]).tolist()
    c_strokes[:, 2] += 1
    c_strokes = sequence.\
        pad_sequences(c_strokes.swapaxes(0, 1), maxlen=MAX_STROKE_COUNT, padding='post').\
        swapaxes(0, 1).\
        astype(np.int64)
    return c_strokes


def split_data(data_dir, tmp_dir, output_dir, nb_samples_for_each_class, train_dev_ratio=0.8):
    """
    :param file_path: 
    :param tmp_dir: 
    :param nb_samples_for_each_class: 
    :param train_dev_ratio: 
    :return: 
    """
    train_file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
    out_df_list = []
    for c_path in train_file_paths:
        c_df = pd.read_csv(c_path, nrows=nb_samples_for_each_class)
        out_df_list += [c_df[['drawing', 'word']]]
    full_df = pd.concat(out_df_list)
    print("data has shape {}".format(full_df.shape))
    # cast labels onto integers
    word_encoder = LabelEncoder()
    word_encoder.fit(full_df['word'])
    print("classes: {}".format(word_encoder.classes_))
    # save to file
    filename = os.path.join(output_dir, ENCODER_NAME)
    joblib.dump(word_encoder, filename)
    full_df['word'] = word_encoder.transform(full_df['word'].values)
    train_df, dev_df = train_test_split(full_df, test_size=1 - train_dev_ratio)
    train_path = os.path.join(tmp_dir, 'train.csv')
    dev_path = os.path.join(tmp_dir, 'dev.csv')
    # write
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    return train_path, dev_path


def get_filenames(mode):
    if mode == 'train':
        return generator_utils.train_data_filenames
    elif mode == 'dev':
        return generator_utils.dev_data_filenames
    else:
        return generator_utils.test_data_filenames


def generate_files(generator, mode, output_dir, nb_shards):
    _fn = get_filenames(mode)
    output_files = _fn('quickdraw', output_dir, nb_shards)
    _generate_files(generator, output_files)


def main(args):
    NB_TRAIN_SHARDS = 1
    NB_DEV_SHARDS = 1
    NB_TEST_SHARDS = 5
    args = vars(args)
    task = args.get('task')
    data_dir = args.get('data_dir')
    test_dir = args.get('test_dir')
    output_dir = args.get('output_dir')
    tmp_dir = args.get('tmp_dir')
    sample_class = int(args.get('sample_class'))
    # split the data into train / dev
    train_path, dev_path = split_data(data_dir, tmp_dir, output_dir, sample_class)
    # generate TF Records for train / dev / test
    # generate TF Records for train / dev / test
    # train
    print('generating samples..')
    generator = generate_samples(task, train_path)
    generate_files(generator, 'train', output_dir, NB_TRAIN_SHARDS)
    print('training complete..')
    # dev
    generator = generate_samples(task, dev_path)
    generate_files(generator, 'dev', output_dir, NB_DEV_SHARDS)
    print('dev complete..')
    # test
    test_path = glob.glob(os.path.join(test_dir, TEST_FILE))[0]
    generator = generate_samples(task, test_path, is_train=False)
    generate_files(generator, 'test', output_dir, NB_TEST_SHARDS)
    print('test complete..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TF Records')
    parser.add_argument('--task', help='task name', required=True, choices=['strokes', 'images'])
    parser.add_argument('--data-dir', help='data directory', required=True)
    parser.add_argument('--test-dir', help='data directory', required=True)
    parser.add_argument('--tmp-dir', help='temp data directory', required=True)
    parser.add_argument('--output-dir', help='output data directory', required=True)
    parser.add_argument('--sample-class', type=int, help='nb samples for each class', required=True)

    args = parser.parse_args()
    main(args)


