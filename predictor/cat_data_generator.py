import numpy as np
import keras
from keras.applications import mobilenet_v2
import os
from PIL import Image

import utils.general
import utils.image
import utils.sampling


class CatDataGenerator(keras.utils.Sequence):
    def __init__(self, path, output_type, include_landmarks=False, batch_size=64, shuffle=True,
                 flip_horizontal=False, rotate=False, rotate_90=False, rotate_n=0,
                 crop=False, crop_scale_balanced_black=False, crop_scale_balanced=False,
                 sampling_method_rotate='random', sampling_method_resize='random'):
        self.path = path
        self.output_type = output_type
        self.include_landmarks = include_landmarks
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.flip_horizontal = flip_horizontal
        self.rotate = rotate
        self.rotate_90 = rotate_90
        self.rotate_n = rotate_n
        self.crop = crop
        self.crop_scale_balanced_black = crop_scale_balanced_black
        self.crop_scale_balanced = crop_scale_balanced
        self.sampling_method_rotate = sampling_method_rotate
        self.sampling_method_resize = sampling_method_resize

        if self.output_type == 'bbox':
            self.output_dim = 4
            if self.include_landmarks:
                self.output_dim += 10
        else:
            self.output_dim = 10

        self.files = [os.path.join(path, f) for f in os.listdir(path) if f[-4:] in ('.jpg', '.bmp', '.gif', '.png')]
        self.indexes = np.arange(len(self.files))

        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x = np.zeros((len(indexes),) + utils.general.img_shape)
        y = np.zeros((len(indexes), self.output_dim))

        for i, idx in enumerate(indexes):
            img_file = self.files[idx]
            img = Image.open(img_file)
            with open(img_file + '.cat', 'r') as cat:
                landmarks = np.array([float(i) for i in cat.readline().split()[1:]]).reshape((-1, 2))

            if self.rotate:
                angle = 360 * np.random.random_sample()
                img, landmarks = utils.image.rotate(img, landmarks, angle, sampling_method=self.sampling_method_rotate)

            if self.rotate_90:
                angle = np.random.choice([0, 90, 180, 270])
                img, landmarks = utils.image.rotate(img, landmarks, angle, sampling_method=self.sampling_method_rotate)

            if self.rotate_n > 0:
                angle = self.rotate_n * (2. * np.random.random_sample() - 1.)
                img, landmarks = utils.image.rotate(img, landmarks, angle, sampling_method=self.sampling_method_rotate)

            if self.output_type == 'bbox':
                if self.crop:
                    bb_crop = utils.sampling.sample_bounding_box(img.size, landmarks)
                    img, landmarks = utils.image.crop(img, landmarks, bb_crop)

                if self.crop_scale_balanced_black:
                    bb_crop = utils.sampling.sample_bounding_box_scale_balanced_black(landmarks)
                    img, landmarks = utils.image.crop(img, landmarks, bb_crop)

                if self.crop_scale_balanced:
                    bb_crop = utils.sampling.sample_bounding_box_scale_balanced(img.size, landmarks)
                    img, landmarks = utils.image.crop(img, landmarks, bb_crop)
            else:
                if self.crop:
                    bb_crop = utils.sampling.sample_bounding_box_landmarks(landmarks)
                    img, landmarks = utils.image.crop(img, landmarks, bb_crop)

            img, landmarks = utils.image.resize(img, landmarks, sampling_method=self.sampling_method_resize)

            if self.flip_horizontal and np.random.random_sample() > 0.5:
                img, landmarks = utils.image.flip(img, landmarks)

            landmarks = np.round(landmarks).astype('int')
            bounding_box = utils.general.get_bounding_box(landmarks)

            if self.output_type == 'bbox':
                if self.include_landmarks:
                    y[i] = np.concatenate((bounding_box, landmarks.flatten()))
                else:
                    y[i] = bounding_box
            else:
                y[i] = landmarks.flatten()
            x[i] = np.asarray(img)

        x = mobilenet_v2.preprocess_input(x)

        return x, y

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
