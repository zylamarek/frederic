import numpy as np
import keras
from keras.applications import mobilenet_v2
import os
from PIL import Image


class CatDataGenerator(keras.utils.Sequence):
    def __init__(self, path, batch_size=64, img_size=224, shuffle=True):
        self.path = path
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.output_dim = 4
        self.img_shape = (self.img_size, self.img_size, 3)

        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((len(indexes),) + self.img_shape)
        y = np.zeros((len(indexes), self.output_dim))

        for i, idx in enumerate(indexes):
            img_file = self.files[idx]
            img = Image.open(img_file)
            with open(img_file + '.cat', 'r') as cat:
                landmarks = np.array([float(i) for i in cat.readline().split()[1:]]).reshape((-1, 2))

            img, landmarks = self._resize_img(img, landmarks)
            landmarks = np.round(landmarks).astype('int')
            y[i] = np.concatenate([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])
            X[i] = np.asarray(img)

        X = mobilenet_v2.preprocess_input(X)

        return X, y

    def _resize_img(self, img, landmarks):
        old_size = img.size
        if old_size != (self.img_size, self.img_size):
            ratio = float(self.img_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            old_img = img.resize(new_size, Image.LANCZOS)

            img = Image.new('RGB', (self.img_size, self.img_size))
            x_diff = (self.img_size - new_size[0]) // 2
            y_diff = (self.img_size - new_size[1]) // 2
            img.paste(old_img, (x_diff, y_diff))

            landmarks *= ratio
            landmarks += np.array((x_diff, y_diff))

        return img, landmarks

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)
