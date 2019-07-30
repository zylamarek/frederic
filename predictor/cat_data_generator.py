import numpy as np
import keras
from keras.applications import mobilenet_v2
import os
from PIL import Image

import utils


class CatDataGenerator(keras.utils.Sequence):
    def __init__(self, path, batch_size=64, shuffle=True, include_landmarks=False, flip_horizontal=False):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.include_landmarks = include_landmarks
        self.flip_horizontal = flip_horizontal

        self.output_dim = 4
        if self.include_landmarks:
            self.output_dim += 10

        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        self.indexes = np.arange(len(self.files))

        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x = np.zeros((len(indexes),) + utils.img_shape)
        y = np.zeros((len(indexes), self.output_dim))

        for i, idx in enumerate(indexes):
            img_file = self.files[idx]
            img = Image.open(img_file)
            with open(img_file + '.cat', 'r') as cat:
                landmarks = np.array([float(i) for i in cat.readline().split()[1:]]).reshape((-1, 2))

            img, landmarks = self._resize_img(img, landmarks)

            if self.flip_horizontal and np.random.random_sample() > 0.5:
                img, landmarks = self._flip_img(img, landmarks)

            landmarks = np.round(landmarks).astype('int')
            bounding_box = np.concatenate([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

            if self.include_landmarks:
                y[i] = np.concatenate((bounding_box, landmarks.flatten()))
            else:
                y[i] = bounding_box
            x[i] = np.asarray(img)

        x = mobilenet_v2.preprocess_input(x)

        return x, y

    @staticmethod
    def _flip_img(img, landmarks):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        landmarks[:, 0] = img.size[0] - landmarks[:, 0]

        # flip eyes and ears landmarks (left becomes right and right becomes left)
        for a, b in ((0, 1), (3, 4)):
            tmp = landmarks[a].copy()
            landmarks[a] = landmarks[b]
            landmarks[b] = tmp

        return img, landmarks

    @staticmethod
    def _resize_img(img, landmarks):
        old_size = img.size
        if old_size != (utils.img_size, utils.img_size):
            ratio = float(utils.img_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            old_img = img.resize(new_size, Image.LANCZOS)

            img = Image.new('RGB', (utils.img_size, utils.img_size))
            x_diff = (utils.img_size - new_size[0]) // 2
            y_diff = (utils.img_size - new_size[1]) // 2
            img.paste(old_img, (x_diff, y_diff))

            landmarks *= ratio
            landmarks += np.array((x_diff, y_diff))

        return img, landmarks

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
