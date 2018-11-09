from data import create_tf_dataset
import datetime
from sklearn.externals import joblib
import pandas as pd
import glob
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import callbacks
import numpy as np


def create_submission_file(model, model_params, model_type):
    print("creating submission file..")
    # load the word encoder
    filepath = glob.glob(os.path.join(model_params.data_path, create_tf_dataset.ENCODER_NAME))
    encoder = joblib.load(filepath[0])
    # load the dataframe to write to
    filepath = glob.glob(os.path.join(model_params.test_path, create_tf_dataset.TEST_FILE))[0]
    submission = pd.read_csv(filepath)
    test_data = create_tf_dataset.generate_samples(model_type, filepath, is_train=False)
    predictions = model.predict(test_data)
    predictions = [encoder.inverse_transform(np.argsort(-1 * c_pred)[:3]) for c_pred in predictions]
    predictions = [' '.join([col.replace(' ', '_') for col in row]) for row in predictions]
    submission['word'] = predictions
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filepath = os.path.join(model_params.test_path, "submission_{}_{}.csv".format(model_type, ts))
    submission[['key_id', 'word']].to_csv(filepath, index=False)
    print("predictions persisted..")


def get_callbacks(model_params, model_type):
    weight_path = "{}/{}_weights.best.hdf5".format(model_params.tmp_data_path, model_type)

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
    callbacks_list = [checkpoint, reduce_lr, early, tensorboard]
    return callbacks_list, weight_path