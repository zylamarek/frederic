import numpy as np
import keras
from keras.applications import mobilenet_v2
import os
from PIL import Image

import utils


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

            if self.rotate_n > 0:
                angle = self.rotate_n * (2. * np.random.random_sample() - 1.)
                img, landmarks = self._rotate(img, landmarks, angle, sampling_method=self.sampling_method_rotate)

            if self.output_type == 'bbox':
                if self.crop:
                    bb_crop = self._sample_bounding_box(img.size, landmarks)
                    img, landmarks = self._crop_bounding_box(img, landmarks, bb_crop)

                if self.crop_scale_balanced_black:
                    bb_crop = self._sample_bounding_box_scale_balanced_black(landmarks)
                    img, landmarks = self._crop_bounding_box(img, landmarks, bb_crop)

                if self.crop_scale_balanced:
                    bb_crop = self._sample_bounding_box_scale_balanced(img.size, landmarks)
                    img, landmarks = self._crop_bounding_box(img, landmarks, bb_crop)
            else:
                if self.crop:
                    bb_crop = self._sample_bounding_box_landmarks(landmarks)
                    img, landmarks = self._crop_bounding_box(img, landmarks, bb_crop)

            img, landmarks = self._resize_img(img, landmarks, sampling_method=self.sampling_method_resize)

            if self.flip_horizontal and np.random.random_sample() > 0.5:
                img, landmarks = self._flip_img(img, landmarks)

            landmarks = np.round(landmarks).astype('int')
            bounding_box = self.get_bounding_box(landmarks)

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

    @staticmethod
    def _rotate(img, landmarks, angle, expand=True, sampling_method='random'):
        if angle in (0, 360):
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
        old_size = img.size

        if angle == 90:
            img = img.transpose(Image.ROTATE_90)
        elif angle == 180:
            img = img.transpose(Image.ROTATE_180)
        elif angle == 270:
            img = img.transpose(Image.ROTATE_270)
        else:
            if sampling_method == 'random':
                sampling_method = np.random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
            img = img.rotate(angle, expand=expand, resample=sampling_method)

        landmarks[:, 0] += (img.size[0] - old_size[0]) / 2
        landmarks[:, 1] += (img.size[1] - old_size[1]) / 2

        return img, landmarks

    @staticmethod
    def _sample_bounding_box(size, landmarks, margin=0.1, bb_min=0.8):
        """
        Samples a bounding box for cropping so that at least 'bb_min' of each dimension of the original bounding box
        is present in the new cropped image.
        """

        # Get old bounding box limited by the size of the image
        bb_old = CatDataGenerator.get_bounding_box(landmarks)
        bb_old = [np.max((0, bb_old[0])),
                  np.max((0, bb_old[1])),
                  np.min((size[0] - 1, bb_old[2])),
                  np.min((size[1] - 1, bb_old[3]))]

        bb_old_size = np.max((bb_old[2] - bb_old[0], bb_old[3] - bb_old[1]))
        img_size_min = int(np.min(size) * (1. + 2. * margin))

        # Sample size of the new bounding box
        bb_crop_size_min = int(bb_old_size * bb_min) + 1
        bb_crop_size_max = np.max((img_size_min - 1, bb_crop_size_min))
        bb_crop_size = np.random.random_integers(low=bb_crop_size_min, high=bb_crop_size_max)

        # Sample x of the starting point of the new bounding box
        bb_crop_start_x_min = np.max((-int(margin * size[0]), bb_old[2] - bb_crop_size))
        bb_crop_start_x_max = np.max((0, bb_old[0] + int((1. - bb_min) * bb_old_size) + 1))
        bb_crop_start_x = np.random.random_integers(low=bb_crop_start_x_min, high=bb_crop_start_x_max)

        # Sample y of the starting point of the new bounding box
        bb_crop_start_y_min = np.max((-int(margin * size[1]), bb_old[3] - bb_crop_size))
        bb_crop_start_y_max = np.max((0, bb_old[1] + int((1. - bb_min) * bb_old_size) + 1))
        bb_crop_start_y = np.random.random_integers(low=bb_crop_start_y_min, high=bb_crop_start_y_max)

        bb_crop = [bb_crop_start_x,
                   bb_crop_start_y,
                   bb_crop_start_x + bb_crop_size,
                   bb_crop_start_y + bb_crop_size]

        return np.array(bb_crop)

    def _sample_bounding_box_scale_balanced(self, size, landmarks):
        """
        Samples a bounding box for cropping so that the distribution of scales in the training data
        is close to uniform and there is much less black borders compared to
        '_sample_bounding_box_scale_balanced_black' method. Works best (the distribution is closest to uniform)
        when run with 'rotate_n' around 15-20 degrees.
        """

        bb_old = CatDataGenerator.get_bounding_box(landmarks)
        bb_old_size = np.max((bb_old[2] - bb_old[0], bb_old[3] - bb_old[1]))

        bb_size_min = bb_old_size
        bb_size_max = 14 * bb_old_size

        # Sample size of the new bounding box based on some statistic that works fine with this particular data
        if np.random.random_sample() > 0.5:
            val = np.random.beta(1.05, 30)
        else:
            val = 1 - np.random.beta(1, 10000)
        bb_crop_size = int(bb_size_min + val * (bb_size_max - bb_size_min))

        bb_crop_start_x = np.random.random_integers(low=np.max((0, bb_old[2] - bb_crop_size)),
                                                    high=np.max((0, bb_old[0] + 1)))
        bb_crop_start_y = np.random.random_integers(low=np.max((0, bb_old[3] - bb_crop_size)),
                                                    high=np.max((0, bb_old[1] + 1)))

        bb_crop = [bb_crop_start_x,
                   bb_crop_start_y,
                   np.min((bb_crop_start_x + bb_crop_size, size[0] - 1)),
                   np.min((bb_crop_start_y + bb_crop_size, size[1] - 1))]

        return np.array(bb_crop)

    def _sample_bounding_box_scale_balanced_black(self, landmarks):
        """
        Samples a bounding box for cropping so that the distribution of scales in the training data is uniform.
        """

        bb_min = 0.9
        bb_old = CatDataGenerator.get_bounding_box(landmarks)
        bb_old_shape = np.array((bb_old[2] - bb_old[0], bb_old[3] - bb_old[1]))
        bb_old_size = np.max(bb_old_shape)
        margin = (1 - bb_min) / 2
        bb_old_min = np.round([bb_old[0] + bb_old_shape[0] * margin,
                               bb_old[1] + bb_old_shape[1] * margin,
                               bb_old[2] - bb_old_shape[0] * margin,
                               bb_old[3] - bb_old_shape[1] * margin])

        scale = np.random.random_sample() * 0.94 + 0.08
        bb_crop_size = int(round(bb_old_size / scale))

        bb_crop_start_x = np.random.random_integers(low=bb_old_min[2] - bb_crop_size,
                                                    high=bb_old_min[0] + 1)
        bb_crop_start_y = np.random.random_integers(low=bb_old_min[3] - bb_crop_size,
                                                    high=bb_old_min[1] + 1)

        bb_crop_end_x = bb_crop_start_x + bb_crop_size
        bb_crop_end_y = bb_crop_start_y + bb_crop_size

        bb_crop = [bb_crop_start_x,
                   bb_crop_start_y,
                   bb_crop_end_x,
                   bb_crop_end_y]

        return np.array(bb_crop)

    def _sample_bounding_box_landmarks(self, landmarks, margin=0.1, random_margin=0.1):
        """
        Samples a bounding box for cropping in landmark prediction training. It takes the ground truth bounding box
        and increases it by 'margin' on each side. Then it samples +- 'random_margin' on each side to simulate
        errors made by the bounding box prediction algorithm. The bounding box is extended to a square, while
        preserving position of the center.
        """

        bb = self.get_bounding_box(landmarks)
        margins = 2 * random_margin * np.random.random_sample(size=4) - random_margin + margin

        bb_size = np.max((bb[2] - bb[0], bb[3] - bb[1]))
        margins *= bb_size
        bb_crop = [bb[0] - margins[0],
                   bb[1] - margins[1],
                   bb[2] + margins[2],
                   bb[3] + margins[3]]

        bb_crop_size = np.max((bb_crop[2] - bb_crop[0], bb_crop[3] - bb_crop[1]))
        bb_crop_center = [(bb_crop[2] + bb_crop[0]) / 2, (bb_crop[3] + bb_crop[1]) / 2]
        bb_crop = [bb_crop_center[0] - bb_crop_size / 2,
                   bb_crop_center[1] - bb_crop_size / 2,
                   bb_crop_center[0] + bb_crop_size / 2,
                   bb_crop_center[1] + bb_crop_size / 2]

        return np.round(bb_crop).astype('int')

    @staticmethod
    def _crop_bounding_box(img, landmarks, bounding_box):
        img = img.crop(bounding_box)
        landmarks -= bounding_box[:2]
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
    def get_bounding_box(landmarks):
        return np.concatenate([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
