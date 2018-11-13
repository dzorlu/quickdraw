
import glob
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import datetime
from sklearn.externals import joblib
import pandas as pd

import cv2
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import callbacks
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical

MAX_STROKE_COUNT = 128
NB_CLASSES = 340
ENCODER_NAME = 'word_encoder.pkl'
TEST_FILE = 'test_simplified.csv'


def _draw_it(raw_strokes, size=128, lw=6, time_color=True):
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
    del raw_strokes
    c_strokes = np.stack(in_strokes)
    del in_strokes

    # replace stroke id with 1 for continue, 2 for new stroke
    c_strokes[:, 2] = [1] + np.diff(c_strokes[:,2]).tolist()
    c_strokes[:, 2] += 1
    c_strokes = sequence.\
        pad_sequences(c_strokes.swapaxes(0, 1), maxlen=MAX_STROKE_COUNT, padding='post').\
        swapaxes(0, 1)
    return c_strokes


def generate_samples_from_file(task, file_path, is_train=True, preprocess=False, batch_size=64):
    if task not in ('strokes', 'image'):
        ValueError("Task not recognized..")
    map_fn = _stack_it if task == 'strokes' else _draw_it
    while True:
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            _map_fn = _stack_it if task == 'strokes' else _draw_it
            chunk['drawing'] = [map_fn(json.loads(draw)) for draw in chunk['drawing'].values]
            batch_x = np.stack(chunk['drawing'].values)
            # TODO: pass the preprocessing as a function that encapsulates input_shape
            if preprocess:
                batch_x = preprocess_input(batch_x).astype(np.float32)
                batch_x = np.repeat(batch_x, 3).reshape(batch_size, 128, 128, 3)
            if is_train:
                batch_y = to_categorical(np.stack(chunk['word'].values), num_classes=NB_CLASSES)
                yield (batch_x, batch_y)
        #TODO: test run needs to be removed!


def create_submission_file(model, model_params, model_type):
    print("creating submission file..")
    # load the word encoder
    filepath = glob.glob(os.path.join(model_params.data_path, ENCODER_NAME))
    encoder = joblib.load(filepath[0])
    # load the dataframe to write to
    filepath = glob.glob(os.path.join(model_params.test_path, TEST_FILE))[0]
    submission = pd.read_csv(filepath)
    test_data = generate_samples_from_file(model_type, filepath, is_train=False)
    predictions = model.predict_generator(test_data, steps=(112200 // model_params.batch_size)+1)
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

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                                  verbose=1, mode='auto', cooldown=2, min_lr=1e-4)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=6)

    tensorboard = callbacks.TensorBoard(log_dir=model_params.tmp_data_path,
                                        histogram_freq=0,
                                        batch_size=model_params.batch_size,
                                        write_graph=True, write_grads=False,
                                        write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None,
                                        embeddings_metadata=None, embeddings_data=None)
    callbacks_list = [checkpoint, reduce_lr, early, tensorboard]
    return callbacks_list, weight_path