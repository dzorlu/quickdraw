# coding=utf-8
from data import read_tf_dataset, create_tf_dataset

import ast
import argparse
import sys
import glob
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.externals import joblib
from keras import callbacks
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv1D, LSTM, Dense, Dropout
from keras.layers import CuDNNLSTM as LSTM
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


NB_CHANNELS = 3


# helper fns
def _get_num_classes():
    clf = joblib.load(FLAGS.model_dir)
    return clf.classes_


def top_3_accuracy(x, y): return top_k_categorical_accuracy(x, y, 3)


def model_fn(params):
    """
    create model function and callbacks given the params
    :return: 
    """

    def _create_cnn_layers(model, model_params):
        for i in range(model_params.num_conv):
            model.add(Conv1D(model_params.nb_conv_filters[i], ()))
            if model_params.dropout:
                model.add(Dropout(model_params.dropout))
        return model

    def _create_lstm_layers(model, model_params):
        for i in range(model_params.num_rnn_layers):
            return_sequences = i + 1 < model_params.num_rnn_layers
            model.add(LSTM(model_params.num_nodes[i], return_sequences=return_sequences))
            if model_params.dropout:
                model.add(Dropout(model_params.dropout))
        return model
    model = Sequential()
    model.add(BatchNormalization(input_shape=params.num_channels))
    model = _create_cnn_layers(model, params)
    model = _create_lstm_layers(model, params)

    model.add(Dense(params.num_classes, activation='softmax'))
    return model


def get_callbacks(model_params):
    weight_path = "{}_weights.best.hdf5".format('stroke_lstm_model')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True, period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,
                                  verbose=1, mode='auto', cooldown=3, min_lr=0.001)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=10)

    tensorboard = callbacks.TensorBoard(log_dir=model_params.tmp_dir, histogram_freq=0,
                                        batch_size=model_params.batch_size,
                                        write_graph=True, write_grads=False,
                                        write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                        embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callbacks_list = [checkpoint, early, reduce_lr, tensorboard]
    return callbacks_list


def create_submission_file(model, model_params):
    # load the word encoder
    filepath = glob.glob(os.path.join(model_params.data_dir, create_tf_dataset.ENCODER_NAME))
    encoder = joblib.load(filepath)
    # load the dataframe
    filepath = glob.glob(os.path.join(model_params.data_dir, create_tf_dataset.TEST_FILE))
    submission = pd.read_csv(filepath)
    test_input_fn = read_tf_dataset.get_iterator(model_params.test_data_path, model_params.batch_size)
    predictions = model.predict(test_input_fn, use_multiprocessing=True)

    predictions = [encoder.inverse_transform[np.argsort(-1 * c_pred)[:3]] for c_pred in predictions]
    submission['word'] = predictions
    submission[['key_id', 'word']].to_csv('submission.csv', index=False)
    print("predictions persisted..")


def main(unused_args):
    model_params = tf.contrib.training.HParams(
        data_path=FLAGS.data_path,
        num_conv_layers=FLAGS.num_conv_layers,
        num_rnn_layers=FLAGS.num_rnn_layers,
        num_nodes=FLAGS.num_nodes,
        batch_size=FLAGS.batch_size,
        nb_conv_filters=ast.literal_eval(FLAGS.nb_conv_filters),
        conv_filter_size=ast.literal_eval(FLAGS.conv_filter_size),
        num_classes=_get_num_classes(),
        num_channels=NB_CHANNELS,
        learning_rate=FLAGS.learning_rate,
        gradient_clipping_norm=FLAGS.gradient_clipping_norm,
        nb_epochs=FLAGS.nb_epochs,
        batch_norm=FLAGS.batch_norm,
        dropout=FLAGS.dropout)
    model = model_fn(model_params)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', top_3_accuracy])
    tf.logging.info(model.summary())
    callbacks = get_callbacks()

    train_input_fn = read_tf_dataset.get_iterator(model_params.data_path, model_params.batch_size)
    eval_input_fn = read_tf_dataset.get_iterator(model_params.data_path, model_params.batch_size)
    history = model.fit_generator(generator=train_input_fn,
                                  validation_data=eval_input_fn,
                                  validation_steps=200,
                                  use_multiprocessing=True,
                                  batch_size=model_params.batch_size,
                                  epochs=model_params.nb_epochs,
                                  # assuming we have 1MM training samples
                                  steps_per_epoch= int(1e3 // model_params.batch_size),
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
      "--tmp_data",
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
      default=3,
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
      default="False",
      help="Whether to enable batch normalization or not.")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=0.0001,
      help="Learning rate used for training.")
  parser.add_argument(
      "--gradient_clipping_norm",
      type=float,
      default=9.0,
      help="Gradient clipping norm used during training.")
  parser.add_argument(
      "--dropout",
      type=float,
      default=0.3,
      help="Dropout used for convolutions and bidi lstm layers.")
  parser.add_argument(
      "--steps",
      type=int,
      default=100000,
      help="Number of training steps.")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=8,
      help="Batch size to use for training/evaluation.")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Path for storing the model checkpoints.")
  parser.add_argument(
      "--nb_epochs",
      type=int,
      default=10,
      help="number of epochs")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
