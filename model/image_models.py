#!/home/deniz/anaconda3/envs/tensorflow_gpuenv/bin/python
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


from model import utils


import argparse


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import DepthwiseConv2D, Activation, Conv2D, \
    BatchNormalization, Reshape, ReLU, AveragePooling2D
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

    base_model = tf.keras.applications.mobilenet.MobileNet(input_shape=(params.input_dim,
                                                                        params.input_dim, 1),
                                                           alpha=1.0,
                                                           depth_multiplier=1,
                                                           dropout=params.dropout,
                                                           include_top=True,
                                                           weights=None,
                                                           classes=NB_CLASSES,
                                                           input_tensor=None,
                                                           pooling=None)
    # x = base_model.output
    # #Freeze lower layers
    # #nb_layers_to_freeze = len(base_model.layers)
    # # for i, layer in enumerate(base_model.layers):
    # #     layer.trainable = True
    # x = DepthwiseConv2D((3, 3),
    #                     padding='same',
    #                     strides=(2, 2),
    #                     use_bias=False,
    #                     name='conv_dw_14')(x)
    # x = BatchNormalization(name='conv_dw_14_bn')(x)
    # x = ReLU(6., name='conv_dw_14_relu')(x)
    # x = Conv2D(128, (1, 1),
    #            padding='same',
    #            use_bias=False,
    #            strides=(1, 1),
    #            name='conv_pw_14')(x)
    # x = BatchNormalization(name='conv_pw_14_bn')(x)
    # x = AveragePooling2D()(x)
    # x = Conv2D(params.num_classes, kernel_size=(1, 1), padding='same', name='conv_15')(x)
    # x = Activation('softmax', name='act_softmax')(x)
    # x = Reshape((params.num_classes,), name='reshape_2')(x)
    # model = Model(inputs=base_model.input, outputs=x)
    base_model.compile(optimizer=Adam(lr=2e-3),
                       loss='categorical_crossentropy',
                       metrics=['categorical_accuracy', top_3_accuracy])
    return base_model


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
    train_path = os.path.join(model_params.tmp_data_path, 'train_images.csv')
    dev_path = os.path.join(model_params.tmp_data_path, 'dev_images.csv')
    preprocess_fn = utils.preprocess_fn(input_shape=model_params.input_dim,
                                        repeat_channels=False)
    train_generator = utils.generate_samples_from_file('image', train_path,
                                                       is_train=True,
                                                       preprocess_fn=preprocess_fn,
                                                       batch_size=model_params.batch_size)
    eval_generator = utils.generate_samples_from_file('image', dev_path,
                                                      preprocess_fn=preprocess_fn,
                                                      is_train=True,
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
      "--input_dim",
      type=int,
      default=128,
      help="Number of training samples.")
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
