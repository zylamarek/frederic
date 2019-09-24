import keras.backend as K
import csv
import numpy as np

IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
L_EYE_LEFT = 0
L_EYE_RIGHT = 1
L_MOUTH = 2
L_EAR_LEFT = 3
L_EAR_RIGHT = 4


def get_loss_fn(output_type, name, iou_and_mse_landmarks_ratio=None):
    if output_type == 'landmarks':
        if not name == 'mse':
            print('Loss_fn "%s" not available in landmarks training. Forcing loss_fn to be "mse".' % name)
        return 'mse'

    loss_fn_map = {'mse': 'mse',
                   'iou': iou_loss}

    if name == 'iou_and_mse_landmarks':
        assert iou_and_mse_landmarks_ratio is not None
        loss_fn = get_iou_and_mse_landmarks_loss(iou_and_mse_landmarks_ratio)
    else:
        loss_fn = loss_fn_map[name]

    return loss_fn


def get_custom_objects(loss_name=None, loss_fn=None):
    custom_objects = {'iou': iou,
                      'iou_loss': iou_loss}
    if loss_name == 'iou_and_mse_landmarks':
        custom_objects['iou_and_mse_landmarks_loss'] = loss_fn
    return custom_objects


def iou(y_true, y_pred):
    """
    Graph version of https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    """

    y_true = K.permute_dimensions(y_true, (1, 0))
    y_pred = K.permute_dimensions(y_pred, (1, 0))

    x_0 = K.max([K.gather(y_true, 0), K.gather(y_pred, 0)], axis=0)
    y_0 = K.max([K.gather(y_true, 1), K.gather(y_pred, 1)], axis=0)
    x_1 = K.min([K.gather(y_true, 2), K.gather(y_pred, 2)], axis=0)
    y_1 = K.min([K.gather(y_true, 3), K.gather(y_pred, 3)], axis=0)

    area_inter = K.clip(x_1 - x_0, 0, None) * K.clip(y_1 - y_0, 0, None)

    area_true = (K.gather(y_true, 2) - K.gather(y_true, 0)) * (K.gather(y_true, 3) - K.gather(y_true, 1))
    area_pred = (K.gather(y_pred, 2) - K.gather(y_pred, 0)) * (K.gather(y_pred, 3) - K.gather(y_pred, 1))

    iou_ = area_inter / (area_true + area_pred - area_inter)

    return K.mean(iou_, axis=-1)


def iou_loss(y_true, y_pred):
    return 1. - iou(y_true, y_pred)


def get_iou_and_mse_landmarks_loss(ratio):
    def iou_and_mse_landmarks_loss(y_true, y_pred):
        iou_ = iou_loss(y_true, y_pred)

        y_true = K.permute_dimensions(y_true, (1, 0))
        y_pred = K.permute_dimensions(y_pred, (1, 0))
        y_true = K.gather(y_true, np.arange(4, 12))
        y_pred = K.gather(y_pred, np.arange(4, 12))
        mse = K.mean(K.square(y_pred - y_true), axis=0)

        return iou_ + mse * ratio

    return iou_and_mse_landmarks_loss


def append_hp_result(path, exp_name, args, history, test_metrics, monitor, mode):
    try:
        with open(path, 'r') as f:
            csv_reader = csv.reader(f, delimiter=';', lineterminator='\n')
            header = next(csv_reader)
    except FileNotFoundError:
        header = ['exp_name'] + list(args) + ['best_ep'] + list(history) + list(test_metrics)
        with open(path, 'w') as f:
            csv_writer = csv.writer(f, delimiter=';', lineterminator='\n')
            csv_writer.writerow(header)

    if mode == 'max':
        best_ep = np.argmax(history[monitor])
    else:
        best_ep = np.argmin(history[monitor])

    pool = {k: v[best_ep] for k, v in history.items()}
    pool['best_ep'] = best_ep
    pool['exp_name'] = exp_name
    pool.update(test_metrics)
    row = [args[h] if h in args.keys() else pool[h] if h in pool else '' for h in header]

    with open(path, 'a') as f:
        csv_writer = csv.writer(f, delimiter=';', lineterminator='\n')
        csv_writer.writerow(row)
