# coding=utf-8
import argparse
import os
import glob
import json
import six

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from PIL import Image, ImageDraw
import tensorflow as tf
from tensor2tensor.data_generators import generator_utils


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if isinstance(v[0], six.integer_types) or isinstance(v[0], np.int64):
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
                       'inputs': [_x.tostring()],
                       'input_shape': [_x.shape[0]]}
    else:
        def _generate_samples():
            for _x in x:
                yield {'inputs': [_x.tostring()],
                       'input_shape': [_x.shape[0]]}
    return _generate_samples()


def _draw_it(strokes, imheight=256, imwidth=256):
    #TODO: Memmory efficient way of doing this?
    """infer a grayscale image from strokes"""
    image = Image.new("P", (256, 256), color=255)
    image_draw = ImageDraw.Draw(image)
    for k, stroke in enumerate(strokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i],
                             stroke[1][i],
                             stroke[0][i+1],
                             stroke[1][i+1]],
                            fill=0, width=5)
    image = image.resize((imheight, imwidth))
    return np.array(image)/255.


def _stack_it(raw_strokes):
    #TODO: PAD SEQUENCES
    """preprocess the string and make 
    a standard Nx3 stroke vector"""
    # unwrap the list
    in_strokes = [(xi, yi, i) for i, (x, y) in enumerate(raw_strokes) for xi, yi in zip(x, y)]
    c_strokes = np.stack(in_strokes)
    # replace stroke id with 1 for continue, 2 for new stroke
    c_strokes[:, 2] = [1] + np.diff(c_strokes[:,2]).tolist()
    c_strokes[:, 2] += 1
    return c_strokes


def split_data(train_file_paths, tmp_dir, nb_samples_for_each_class, train_dev_ratio=0.8):
    #TODO: Save word encoder
    """
    
    :param file_path: 
    :param tmp_dir: 
    :param nb_samples_for_each_class: 
    :param train_dev_ratio: 
    :return: 
    """
    out_df_list = []
    for c_path in train_file_paths:
        c_df = pd.read_csv(c_path, nrows=nb_samples_for_each_class)
        out_df_list += [c_df[['drawing', 'word']]]
    full_df = pd.concat(out_df_list)
    # cast labels onto integers
    word_encoder = LabelEncoder()
    word_encoder.fit(full_df['word'])
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
    NB_TRAIN_SHARDS = 20
    NB_DEV_SHARDS = 1
    NB_TEST_SHARDS = 5
    args = vars(args)
    task = args.get('task')
    data_dir = args.get('data_dir')
    output_dir = args.get('output_dir')
    tmp_dir = args.get('tmp_dir')
    sample_class = int(args.get('sample_class'))
    # split the data into train / dev
    train_file_paths = glob.glob(os.path.join(data_dir, 'train_simplified', '*.csv'))
    train_path, dev_path = split_data(train_file_paths, tmp_dir, sample_class)

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
    test_path = glob.glob(os.path.join(data_dir, 'test_simplified.csv'))[0]
    generator = generate_samples(task, test_path, is_train=False)
    generate_files(generator, 'test', output_dir, NB_TEST_SHARDS)
    print('test complete..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TF Records')
    parser.add_argument('--task', help='task name', required=True, choices=['strokes', 'images'])
    parser.add_argument('--data-dir', help='data directory', required=True)
    parser.add_argument('--tmp-dir', help='temp data directory', required=True)
    parser.add_argument('--output-dir', help='output data directory', required=True)
    parser.add_argument('--sample-class', type=int, help='nb samples for each class', required=True)

    args = parser.parse_args()
    main(args)


