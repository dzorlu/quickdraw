#!/home/deniz/anaconda3/envs/tensorflow_gpuenv/bin/python
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


from model import utils


import argparse


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Reshape
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.metrics import top_k_categorical_accuracy


NB_CHANNELS = 3
NB_CLASSES = 340


def top_3_accuracy(x, y): return top_k_categorical_accuracy(x, y, 3)


def model_fn(params):
    """
    create model function and callbacks given the params
    :return:
    """
    if params.input_dim not in [224, 128]:
        ValueError('hip to be square..')
    #TODO: Preproess input
    base_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(params.input_dim, params.input_dim, 3),
                                                           alpha=1.0,
                                                           depth_multiplier=1,
                                                           dropout=params.dropout,
                                                           include_top=False,
                                                           weights='imagenet',
                                                           input_tensor=None,
                                                           pooling=None)
    x = base_model.output
    #Freeze lower layers
    for layer in base_model.layers:
        layer.trainable = False
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(params.dropout, name='dropout')(x)
    x = Conv2D(params.num_classes, kernel_size=(1,1), padding='same', name='conv_1')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((params.num_classes,), name='reshape_2')(x)
    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', top_3_accuracy])
    return model


def main(args):
    model_params = tf.contrib.training.HParams(
        data_path=FLAGS.data_path,
        test_path=FLAGS.test_path,
        tmp_data_path=FLAGS.tmp_data_path,
        batch_size=FLAGS.batch_size,
        num_classes=NB_CLASSES,
        num_channels=NB_CHANNELS,
        nb_epochs=FLAGS.nb_epochs,
        nb_samples=FLAGS.nb_samples,
        nb_eval_samples=FLAGS.nb_eval_samples,
        input_dim=FLAGS.input_dim,
        dropout=FLAGS.dropout)
    print(model_params)
    model = model_fn(model_params)

    tf.logging.info(model.summary())
    callbacks, weight_path = utils.get_callbacks(model_params, 'image')
    train_path = os.path.join(model_params.tmp_data_path, 'train_image.csv')
    dev_path = os.path.join(model_params.tmp_data_path, 'dev_image.csv')
    train_generator = utils.generate_samples_from_file('image', train_path, is_train=True,
                                                       batch_size=model_params.batch_size)
    eval_generator = utils.generate_samples_from_file('image', dev_path, is_train=True,
                                                      batch_size=model_params.batch_size)

    history = model.fit_generator(train_generator,
                                  validation_data=eval_generator,
                                  validation_steps=500,
                                  epochs=model_params.nb_epochs,
                                  steps_per_epoch=int(model_params.nb_samples) // model_params.batch_size,
                                  callbacks=callbacks)
    # evaluate
    model.load_weights(weight_path)
    eval_res = model.evaluate_generator(eval_generator, steps=model_params.nb_eval_samples // model_params.batch_size)
    print('Accuracy: %2.1f%%, Top 3 Accuracy %2.1f%%' % (100 * eval_res[1], 100 * eval_res[2]))
    # submit
    utils.create_submission_file(model, model_params, 'image')
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
      "--nb_eval_samples",
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
