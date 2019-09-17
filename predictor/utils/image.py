import numpy as np
from PIL import Image

import utils.general


def rotate(img, landmarks, angle, expand=True, sampling_method='random'):
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


def resize(img, landmarks, sampling_method='random'):
    old_size = img.size
    if old_size != (utils.general.img_size, utils.general.img_size):
        ratio = float(utils.general.img_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        if sampling_method == 'random':
            sampling_method = np.random.choice([Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING,
                                                Image.BICUBIC, Image.LANCZOS])
        old_img = img.resize(new_size, sampling_method)

        img = Image.new('RGB', (utils.general.img_size, utils.general.img_size))
        x_diff = (utils.general.img_size - new_size[0]) // 2
        y_diff = (utils.general.img_size - new_size[1]) // 2
        img.paste(old_img, (x_diff, y_diff))

        landmarks *= ratio
        landmarks += np.array((x_diff, y_diff))

    return img, landmarks


def flip(img, landmarks):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    landmarks[:, 0] = img.size[0] - landmarks[:, 0]

    # flip eyes and ears landmarks (left becomes right and right becomes left)
    for a, b in ((0, 1), (3, 4)):
        tmp = landmarks[a].copy()
        landmarks[a] = landmarks[b]
        landmarks[b] = tmp

    return img, landmarks


def crop(img, landmarks, bounding_box):
    img = img.crop(bounding_box)
    landmarks -= bounding_box[:2]
    return img, landmarks
