import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from collections import defaultdict

import context
from frederic.predictor import Predictor
import frederic.utils.image
from frederic.utils.general import L_EYE_LEFT, L_EYE_RIGHT, L_MOUTH, L_EAR_LEFT, L_EAR_RIGHT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=os.path.join('..', '..', 'cat-dataset', 'data', 'clean', 'validation'))
    args = parser.parse_args()

    predictor = Predictor()

    output_dir = os.path.join('output', datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S"))
    os.makedirs(output_dir, exist_ok=True)

    mses = defaultdict(list)
    for img_filename in tqdm([f for f in os.listdir(args.data_path) if f[-4:] in ('.jpg', '.bmp', '.gif', '.png')]):
        img_path = os.path.join(args.data_path, img_filename)
        img, landmarks_truth = frederic.utils.image.load(img_path)
        landmarks_predicted = predictor.predict(img)


        def get_mse_normalized(a, b):
            return np.mean(np.square((landmarks_truth[a: b + 1] - landmarks_predicted[a: b + 1]) / np.max(img.size)))


        mse = np.mean(np.square(landmarks_truth - landmarks_predicted))
        mses['all'].append(mse)
        mses['all normalized'].append(get_mse_normalized(L_EYE_RIGHT, L_EAR_LEFT))
        mses['eyes normalized'].append(get_mse_normalized(L_EYE_RIGHT, L_EYE_LEFT))
        mses['mouth normalized'].append(get_mse_normalized(L_MOUTH, L_MOUTH))
        mses['ears normalized'].append(get_mse_normalized(L_EAR_RIGHT, L_EAR_LEFT))

        output_filename = '%.9f_%s' % (np.sqrt(mse), img_filename)
        output_path = os.path.join(output_dir, output_filename)
        frederic.utils.image.save_with_landmarks(img, landmarks_truth, landmarks_predicted, output_path)
        predictor.save_landmarks(landmarks_predicted, output_path + '.cat')

    for name, vals in mses.items():
        print('rmse %s: %.7f' % (name, np.sqrt(np.mean(vals))))
