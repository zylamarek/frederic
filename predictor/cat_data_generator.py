import numpy as np
import keras
from keras.applications import mobilenet_v2
import os
from PIL import Image

import utils


class CatDataGenerator(keras.utils.Sequence):
    def __init__(self, path, batch_size=64, shuffle=True, include_landmarks=False,
                 flip_horizontal=False, rotate=False, rotate_90=False,
                 sampling_method_rotate='random', sampling_method_resize='random'):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.include_landmarks = include_landmarks

        self.flip_horizontal = flip_horizontal
        self.rotate = rotate
        self.rotate_90 = rotate_90
        self.sampling_method_rotate = sampling_method_rotate
        self.sampling_method_resize = sampling_method_resize

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

            if self.rotate:
                angle = 360 * np.random.random_sample()
                img, landmarks = self._rotate(img, landmarks, angle, sampling_method=self.sampling_method_rotate)

            if self.rotate_90:
                angle = np.random.choice([0, 90, 180, 270])
                img, landmarks = self._rotate(img, landmarks, angle, sampling_method=self.sampling_method_rotate)

            img, landmarks = self._resize_img(img, landmarks, sampling_method=self.sampling_method_resize)

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
    def _rotate(img, landmarks, angle, sampling_method='random'):
        if angle == 0:
            return img, landmarks

        radians = np.radians(angle)
        offset_x, offset_y = img.size[0] / 2, img.size[1] / 2
        adjusted_x = landmarks[:, 0] - offset_x
        adjusted_y = landmarks[:, 1] - offset_y
        cos_rad = np.cos(radians)
        sin_rad = np.sin(radians)
        qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
        qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
        landmarks = np.array([qx, qy]).T

        if angle == 90:
            img = img.transpose(Image.ROTATE_90)
        elif angle == 180:
            img = img.transpose(Image.ROTATE_180)
        elif angle == 270:
            img = img.transpose(Image.ROTATE_270)
        else:
            if sampling_method == 'random':
                sampling_method = np.random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
            img = img.rotate(angle, resample=sampling_method)

        return img, landmarks

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
    def _resize_img(img, landmarks, sampling_method='random'):
        old_size = img.size
        if old_size != (utils.img_size, utils.img_size):
            ratio = float(utils.img_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            if sampling_method == 'random':
                sampling_method = np.random.choice([Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING,
                                                    Image.BICUBIC, Image.LANCZOS])
            old_img = img.resize(new_size, sampling_method)

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
