import numpy as np
from keras.models import load_model
from PIL import Image
from keras.applications import mobilenet_v2
from keras.utils.data_utils import get_file

import frederic.utils.general
import frederic.utils.image

BASE_MODEL_URL = 'https://docs.google.com/uc?export=download&id='
BBOX_MODEL_ID = '1I3ABL4Ykg5e_mnt0-yN2MahJ5TK8Onyt'
LANDMARKS_MODEL_ID = '11LGQoVWsxn1gq3CRQbWLXzDP0vWKWB72'


class Predictor:
    def __init__(self, bbox_model_path=None, landmarks_model_path=None):
        if bbox_model_path is None:
            bbox_model_path = get_file('frederic_bbox.h5', BASE_MODEL_URL + BBOX_MODEL_ID, cache_subdir='models')
        if landmarks_model_path is None:
            landmarks_model_path = get_file('frederic_landmarks.h5', BASE_MODEL_URL + LANDMARKS_MODEL_ID,
                                            cache_subdir='models')

        dummy_loss_fn = frederic.utils.general.get_loss_fn('bbox', 'iou_and_mse_landmarks', 1e-5)
        custom_objects = frederic.utils.general.get_custom_objects('iou_and_mse_landmarks', dummy_loss_fn)
        self.bbox_model = load_model(bbox_model_path, custom_objects=custom_objects)
        self.landmarks_model = load_model(landmarks_model_path, custom_objects=custom_objects)

    def predict(self, img):
        img_bbox, img_landmarks = img.copy(), img.copy()

        # predict bounding box
        img_bbox, _ = frederic.utils.image.resize(img_bbox, 0, sampling_method=Image.LANCZOS)
        x = np.expand_dims(mobilenet_v2.preprocess_input(np.asarray(img_bbox)), axis=0)
        bbox = self.bbox_model.predict(x, verbose=0)[0, :4]

        # scale and translate predicted bounding box
        bbox = frederic.utils.image.postprocess_bounding_box(bbox, img.size)

        # predict landmarks inside predicted bounding box
        img_landmarks, _ = frederic.utils.image.crop(img_landmarks, 0, bbox)
        img_landmarks, _ = frederic.utils.image.resize(img_landmarks, 0, sampling_method=Image.LANCZOS)
        x = np.expand_dims(mobilenet_v2.preprocess_input(np.asarray(img_landmarks)), axis=0)
        y_pred = self.landmarks_model.predict(x, verbose=0)[0]

        # scale and translate predicted landmarks
        bb_size = np.max((bbox[2] - bbox[0], bbox[3] - bbox[1]))
        ratio = bb_size / frederic.utils.general.IMG_SIZE
        predicted_landmarks = (y_pred * ratio).reshape((-1, 2)) + bbox[:2]

        return predicted_landmarks
