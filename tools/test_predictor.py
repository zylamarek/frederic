import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from collections import defaultdict

import context
from frederic import Predictor
import frederic.utils.image
from frederic.utils.general import L_EYE_LEFT, L_EYE_RIGHT, L_MOUTH, L_EAR_LEFT, L_EAR_RIGHT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=os.path.join('..', '..', 'cat-dataset', 'data', 'clean', 'test'))
    args = parser.parse_args()

    predictor = Predictor()

    output_dir = os.path.join('output', datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    os.makedirs(output_dir)

    metrics = defaultdict(list)
    for img_filename in tqdm([f for f in os.listdir(args.data_path) if f[-4:] in ('.jpg', '.bmp', '.gif', '.png')]):
        img_path = os.path.join(args.data_path, img_filename)
        img, landmarks_truth = frederic.utils.image.load(img_path)
        landmarks_predicted = predictor.predict(img)

        bb_truth = frederic.utils.image.get_bounding_box(landmarks_truth)
        face_size = np.max(np.diff(bb_truth.reshape((-1, 2)), axis=0))


        def get_mape(a, b):
            return np.mean(np.abs((landmarks_truth[a: b + 1] - landmarks_predicted[a: b + 1]) / face_size * 100.))


        err = landmarks_truth - landmarks_predicted
        metrics['mae'].append(np.mean(np.abs(err)))
        metrics['mse'].append(np.mean(np.square(err)))
        metrics['mspe'].append(np.mean(np.square(err / face_size * 100.)))
        mape = np.mean(np.abs(err / face_size * 100.))
        metrics['mape'].append(mape)
        metrics['mape eyes'].append(get_mape(L_EYE_RIGHT, L_EYE_LEFT))
        metrics['mape mouth'].append(get_mape(L_MOUTH, L_MOUTH))
        metrics['mape ears'].append(get_mape(L_EAR_RIGHT, L_EAR_LEFT))

        output_filename = '%.9f_%s' % (mape, img_filename)
        output_path = os.path.join(output_dir, output_filename)
        frederic.utils.image.save_with_landmarks(img, output_path, landmarks_truth, landmarks_predicted)
        frederic.utils.image.save_landmarks(landmarks_predicted, output_path + '.cat')

    for name, vals in metrics.items():
        print('%s:\t%.2f' % (name, np.mean(vals)))
        if name.startswith('ms'):
            print('r%s:\t%.2f' % (name, np.sqrt(np.mean(vals))))
