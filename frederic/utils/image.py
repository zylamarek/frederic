import numpy as np
from PIL import Image

from .general import IMG_SIZE, L_EYE_LEFT, L_EYE_RIGHT, L_EAR_LEFT, L_EAR_RIGHT


def load(path):
    img = Image.open(path)
    with open(path + '.cat', 'r') as cat:
        landmarks = np.array([float(i) for i in cat.readline().split()[1:]]).reshape((-1, 2))
    return img, landmarks


def get_bounding_box(landmarks):
    return np.concatenate([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])


def postprocess_bounding_box(bb, img_size, margin=0.1):
    ratio = float(IMG_SIZE) / max(img_size)
    new_size = tuple(int(x * ratio) for x in img_size)
    x_diff = (IMG_SIZE - new_size[0]) // 2
    y_diff = (IMG_SIZE - new_size[1]) // 2
    bb -= np.array((x_diff, y_diff, x_diff, y_diff))
    bb /= ratio

    bb_size = np.max((bb[2] - bb[0], bb[3] - bb[1]))
    margin *= bb_size
    bb_crop = [bb[0] - margin,
               bb[1] - margin,
               bb[2] + margin,
               bb[3] + margin]

    bb_crop_size = np.max((bb_crop[2] - bb_crop[0], bb_crop[3] - bb_crop[1]))
    bb_crop_center = [(bb_crop[2] + bb_crop[0]) / 2, (bb_crop[3] + bb_crop[1]) / 2]
    bb_crop = [bb_crop_center[0] - bb_crop_size / 2,
               bb_crop_center[1] - bb_crop_size / 2,
               bb_crop_center[0] + bb_crop_size / 2,
               bb_crop_center[1] + bb_crop_size / 2]

    return np.round(bb_crop).astype('int')


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
    if old_size != (IMG_SIZE, IMG_SIZE):
        ratio = float(IMG_SIZE) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        if sampling_method == 'random':
            sampling_method = np.random.choice([Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING,
                                                Image.BICUBIC, Image.LANCZOS])
        old_img = img.resize(new_size, sampling_method)

        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        x_diff = (IMG_SIZE - new_size[0]) // 2
        y_diff = (IMG_SIZE - new_size[1]) // 2
        img.paste(old_img, (x_diff, y_diff))

        landmarks *= ratio
        landmarks += np.array((x_diff, y_diff))

    return img, landmarks


def flip(img, landmarks):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    landmarks[:, 0] = img.size[0] - landmarks[:, 0]

    # flip eyes and ears landmarks (left becomes right and right becomes left)
    for a, b in ((L_EYE_LEFT, L_EYE_RIGHT), (L_EAR_LEFT, L_EAR_RIGHT)):
        tmp = landmarks[a].copy()
        landmarks[a] = landmarks[b]
        landmarks[b] = tmp

    return img, landmarks


def crop(img, landmarks, bounding_box):
    img = img.crop(bounding_box)
    landmarks -= bounding_box[:2]
    return img, landmarks
