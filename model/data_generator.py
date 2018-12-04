

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
import pandas as pd

CHUNK_SIZE = 1e6
MAX_STROKE_COUNT = 128
BASE_SIZE = 256
NB_CLASSES = 340
ENCODER_NAME = 'word_encoder.pkl'
TEST_FILE = 'test_simplified.csv'

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, task,
                 file_path,
                 nb_files,
                 batch_size,
                 input_dim,
                 nb_samples,
                 is_train=True,
                 repeat_channels=False):
        if task not in ('strokes', 'image'):
            ValueError("Task not recognized..")
        self.task = task
        self.file_path = file_path
        self.is_train = is_train
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.nb_chunks = nb_samples // CHUNK_SIZE
        self.repeat_channels = repeat_channels
        self.map_fn = self._stack_it if task == 'strokes' else self._draw_it
        # randomize the files
        self.file_ixs = np.random.choice(self.nb_files, self.nb_chunks, replace=False)

    def on_epoch_end(self):
        self.file_ixs = np.random.choice(self.nb_files, self.nb_chunks, replace=False)

    @staticmethod
    def _draw_it(raw_strokes, lw=6, time_color=True):
        img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
        for t, stroke in enumerate(raw_strokes):
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(t, 10) * 13 if time_color else 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                             (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
        return img

    @staticmethod
    def _stack_it(raw_strokes):
        """preprocess the string and make
        a standard Nx3 stroke vector"""
        # unwrap the list
        in_strokes = [(xi, yi, i) for i, (x, y) in enumerate(raw_strokes) for xi, yi in zip(x, y)]
        del raw_strokes
        c_strokes = np.stack(in_strokes)
        del in_strokes

        # replace stroke id with 1 for continue, 2 for new stroke
        c_strokes[:, 2] = [1] + np.diff(c_strokes[:, 2]).tolist()
        c_strokes[:, 2] += 1
        c_strokes = sequence. \
            pad_sequences(c_strokes.swapaxes(0, 1), maxlen=MAX_STROKE_COUNT, padding='post'). \
            swapaxes(0, 1)
        return c_strokes

    def _preprocess_fn(self, _x):
        _x = cv2.resize(_x, (self.input_dim, self.input_dim))
        if self.repeat_channels:
            _x = np.repeat(_x, 3).reshape(self.input_dim, self.input_dim, 3)
        else:
            _x = np.expand_dims(_x, -1)
        return _x

    def __len__(self):
        _nb_samples = self.nb_chunks * CHUNK_SIZE
        return int(np.floor(_nb_samples / self.batch_size))

    def _get_filename(self, idx):
        # get the # on the file.
        _ix_end = (idx+1)*self.batch_size
        _file_ix = _ix_end // CHUNK_SIZE
        return 'train_{}_{}.csv'.format(self.task, _file_ix )

    def __getitem__(self, idx):
        _filename = self._get_filename(idx)
        for chunk in pd.read_csv(_filename, chunksize=self.batch_size):
            chunk['drawing'] = [map_fn(json.loads(draw)) for draw in chunk['drawing'].values]
            if preprocess_fn:
                _vals = chunk['drawing'].apply(self.preprocess_fn).values
            else:
                _vals = chunk['drawing'].values
            batch_x = np.stack(_vals)
            if preprocess_fn:
                batch_x = preprocess_input(batch_x).astype(np.float32)
            if is_train:
                batch_y = to_categorical(np.stack(chunk['word'].values), num_classes=NB_CLASSES)
                yield (batch_x, batch_y)
            else:
                yield batch_x