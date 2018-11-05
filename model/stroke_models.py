#!/home/deniz/anaconda3/envs/tensorflow_gpuenv/bin/python
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from data import read_tf_dataset, create_tf_dataset

import ast
import argparse
import glob
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.externals import joblib
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.layers import CuDNNLSTM as LSTM
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


NB_CHANNELS = 3
NB_CLASSES = 340


def top_3_accuracy(x, y): return top_k_categorical_accuracy(x, y, 3)


def model_fn(params):
    """
    create model function and callbacks given the params
    :return: 
    """

    def _create_cnn_layers(model, model_params):
        for i in range(model_params.num_conv_layers):
            model.add(Conv1D(model_params.nb_conv_filters[i],
                             (model_params.conv_filter_size[i])))
            if model_params.dropout:
                model.add(Dropout(model_params.dropout))
        return model

    def _create_lstm_layers(model, model_params):
        for i in range(model_params.num_rnn_layers):
            return_sequences = i + 1 < model_params.num_rnn_layers
            model.add(LSTM(model_params.num_nodes,
                           return_sequences=return_sequences))
            if model_params.dropout:
                model.add(Dropout(model_params.dropout))
        return model
    # model = Sequential()
    # model.add(BatchNormalization(input_shape=(create_tf_dataset.MAX_STROKE_COUNT, params.num_channels)))
    # model = _create_cnn_layers(model, params)
    # model = _create_lstm_layers(model, params)
    # model.add(Dense(512))
    # model.add(Dropout(params.dropout))
    # model.add(Dense(params.num_classes, activation='softmax'))
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['categorical_accuracy', top_3_accuracy])    # model = Sequential()
    # model.add(BatchNormalization(input_shape=(create_tf_dataset.MAX_STROKE_COUNT, params.num_channels)))
    # model = _create_cnn_layers(model, params)
    # model = _create_lstm_layers(model, params)
    # model.add(Dense(512))
    # model.add(Dropout(params.dropout))
    # model.add(Dense(params.num_classes, activation='softmax'))
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy',
    #               metrics=['categorical_accuracy', top_3_accuracy])
    stroke_read_model = Sequential()
    stroke_read_model.add(BatchNormalization(input_shape=(None, 3)))
    stroke_read_model.add(Conv1D(48, (5,)))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Conv1D(64, (5,)))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Conv1D(96, (3,)))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(LSTM(128, return_sequences=True))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(LSTM(128, return_sequences=False))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Dense(512))
    stroke_read_model.add(Dropout(0.3))
    stroke_read_model.add(Dense(340, activation='softmax'))
    adam = tf.keras.optimizers.Adam(lr=0.01)
    stroke_read_model.compile(optimizer=adam,
                              loss='categorical_crossentropy',
                              metrics=['categorical_accuracy', top_3_accuracy])
    stroke_read_model.summary()
    return stroke_read_model


def get_callbacks(model_params):
    weight_path = "{}/{}_weights.best.hdf5".format(model_params.tmp_data_path, 'stroke_lstm_model')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                  verbose=1, mode='auto', cooldown=3, min_lr=0.001)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=10)

    tensorboard = callbacks.TensorBoard(log_dir=model_params.tmp_data_path,
                                        histogram_freq=0,
                                        batch_size=model_params.batch_size,
                                        write_graph=True, write_grads=False,
                                        write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None,
                                        embeddings_metadata=None, embeddings_data=None)
    #callbacks_list = [checkpoint, reduce_lr, early, tensorboard]
    callbacks_list = [checkpoint, tensorboard]

    return callbacks_list


def create_submission_file(model, model_params):
    # load the word encoder
    filepath = glob.glob(os.path.join(model_params.data_path, create_tf_dataset.ENCODER_NAME))
    encoder = joblib.load(filepath[0])
    # load the dataframe to write to
    filepath = glob.glob(os.path.join(model_params.test_path, create_tf_dataset.TEST_FILE))
    print(filepath)
    submission = pd.read_csv(filepath[0])
    # initiate the iterator
    _iter = read_tf_dataset.Iter(model_params.data_path, 'test', model_params.batch_size)
    #test_input_fn = read_tf_dataset.get_iterator(model_params.data_path, 'test', model_params.batch_size)
    print(type(_iter))
    predictions = model.predict_generator(_iter, steps=1)

    predictions = [encoder.inverse_transform[np.argsort(-1 * c_pred)[:3]] for c_pred in predictions]
    submission['word'] = predictions
    submission[['key_id', 'word']].to_csv('submission.csv', index=False)
    print("predictions persisted..")


def main(args):
    model_params = tf.contrib.training.HParams(
        data_path=FLAGS.data_path,
        test_path = FLAGS.test_path,
        tmp_data_path=FLAGS.tmp_data_path,
        num_conv_layers=FLAGS.num_conv_layers,
        num_rnn_layers=FLAGS.num_rnn_layers,
        num_nodes=FLAGS.num_nodes,
        batch_size=FLAGS.batch_size,
        nb_conv_filters=ast.literal_eval(FLAGS.nb_conv_filters),
        conv_filter_size=ast.literal_eval(FLAGS.conv_filter_size),
        num_classes=NB_CLASSES,
        num_channels=NB_CHANNELS,
        nb_epochs=FLAGS.nb_epochs,
        nb_samples=FLAGS.nb_samples,
        batch_norm=FLAGS.batch_norm,
        dropout=FLAGS.dropout)
    print(model_params)
    model = model_fn(model_params)

    tf.logging.info(model.summary())
    callbacks = get_callbacks(model_params)

    train_input_fn = read_tf_dataset.get_iterator(model_params.data_path, 'train', model_params.batch_size)
    eval_input_fn = read_tf_dataset.get_iterator(model_params.data_path, 'dev', model_params.batch_size)

    history = model.fit(train_input_fn,
                        validation_data=eval_input_fn,
                        validation_steps=200,
                        epochs=model_params.nb_epochs,
                        steps_per_epoch=int(model_params.nb_samples) // model_params.batch_size,
                        callbacks=callbacks)

    create_submission_file(model, model_params)
    return history


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--data_path",
      type=str,
      default="",
      help="Path to training/eval/test data (tf.Example in TFRecord format)")
  parser.add_argument(
      "--test_path",
      type=str,
      default="",
      help="Path to read/write submission file)")
  parser.add_argument(
      "--tmp_data_path",
      type=str,
      default="",
      help="Path to temp data")
  parser.add_argument(
      "--num_conv_layers",
      type=int,
      default=3,
      help="Number of conv neural network layers.")
  parser.add_argument(
      "--num_rnn_layers",
      type=int,
      default=2,
      help="Number of recurrent neural network layers.")
  parser.add_argument(
      "--num_nodes",
      type=int,
      default=128,
      help="Number of node per recurrent network layer.")
  parser.add_argument(
      "--nb_conv_filters",
      type=str,
      default="[48, 64, 96]",
      help="Number of conv layers along with number of filters per layer.")
  parser.add_argument(
      "--conv_filter_size",
      type=str,
      default="[5, 5, 3]",
      help="Length of the convolution filters.")
  parser.add_argument(
      "--batch_norm",
      type="bool",
      default="True",
      help="Whether to enable batch normalization or not.")
  parser.add_argument(
      "--dropout",
      type=float,
      default=0.3,
      help="Dropout used for convolutions and bidi lstm layers.")
  parser.add_argument(
      "--nb_samples",
      type=int,
      default=5,
      help="Number of training samples.")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=64,
      help="Batch size to use for training/evaluation.")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Path for storing the model checkpoints.")
  parser.add_argument(
      "--nb_epochs",
      type=int,
      default=50,
      help="number of epochs")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
