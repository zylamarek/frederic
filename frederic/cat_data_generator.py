import numpy as np
import keras
from keras.applications import mobilenet_v2
import os

import frederic.utils.general
import frederic.utils.image
import frederic.utils.sampling


class CatDataGenerator(keras.utils.Sequence):
    def __init__(self, path, output_type, include_landmarks=False, batch_size=64, shuffle=True,
                 flip_horizontal=False, rotate=False, rotate_90=False, rotate_n=0,
                 crop=False, crop_scale_balanced_black=False, crop_scale_balanced=False,
                 sampling_method_rotate='random', sampling_method_resize='random',
                 crop_landmarks_margin=0.1, crop_landmarks_random_margin=0.1):
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
        self.crop_landmarks_margin = crop_landmarks_margin
        self.crop_landmarks_random_margin = crop_landmarks_random_margin

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
        x = np.zeros((len(indexes),) + frederic.utils.general.IMG_SHAPE)
        y = np.zeros((len(indexes), self.output_dim))

        for i, idx in enumerate(indexes):
            img, landmarks = frederic.utils.image.load(self.files[idx])
            img, landmarks = self._augment(img, landmarks)
            img, landmarks = frederic.utils.image.resize(img, landmarks, sampling_method=self.sampling_method_resize)
            landmarks = np.round(landmarks).astype('int')

            if self.output_type == 'bbox':
                bounding_box = frederic.utils.image.get_bounding_box(landmarks)
                if self.include_landmarks:
                    y[i] = np.concatenate((bounding_box, landmarks.flatten()))
                else:
                    y[i] = bounding_box
            else:
                y[i] = landmarks.flatten()
            x[i] = np.asarray(img)

        x = mobilenet_v2.preprocess_input(x)

        return x, y

    def _augment(self, img, landmarks):
        if self.rotate:
            angle = 360 * np.random.random_sample()
            img, landmarks = frederic.utils.image.rotate(img, landmarks, angle,
                                                         sampling_method=self.sampling_method_rotate)

        if self.rotate_90:
            angle = np.random.choice([0, 90, 180, 270])
            img, landmarks = frederic.utils.image.rotate(img, landmarks, angle,
                                                         sampling_method=self.sampling_method_rotate)

        if self.rotate_n > 0:
            angle = self.rotate_n * (2. * np.random.random_sample() - 1.)
            img, landmarks = frederic.utils.image.rotate(img, landmarks, angle,
                                                         sampling_method=self.sampling_method_rotate)

        if self.output_type == 'bbox':
            if self.crop:
                bb_crop = frederic.utils.sampling.sample_bounding_box(img.size, landmarks)
                img, landmarks = frederic.utils.image.crop(img, landmarks, bb_crop)

            if self.crop_scale_balanced_black:
                bb_crop = frederic.utils.sampling.sample_bounding_box_scale_balanced_black(landmarks)
                img, landmarks = frederic.utils.image.crop(img, landmarks, bb_crop)

            if self.crop_scale_balanced:
                bb_crop = frederic.utils.sampling.sample_bounding_box_scale_balanced(img.size, landmarks)
                img, landmarks = frederic.utils.image.crop(img, landmarks, bb_crop)
        else:
            if self.crop:
                bb_crop = frederic.utils.sampling.sample_bounding_box_landmarks(landmarks, self.crop_landmarks_margin,
                                                                                self.crop_landmarks_random_margin)
                img, landmarks = frederic.utils.image.crop(img, landmarks, bb_crop)

        if self.flip_horizontal and np.random.random_sample() > 0.5:
            img, landmarks = frederic.utils.image.flip(img, landmarks)

        return img, landmarks

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
