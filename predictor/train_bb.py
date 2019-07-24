from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import datetime
from keras.applications import mobilenet_v2
from keras.layers import Dense
from keras.models import Model, load_model
import argparse
import keras
import keras.backend as K
import os
import csv
import numpy as np

from cat_data_generator import CatDataGenerator


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

    iou = area_inter / (area_true + area_pred - area_inter)

    return iou


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--units', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--ReduceLROnPlateau_factor', default=0.5, type=float)
    parser.add_argument('--ReduceLROnPlateau_patience', default=5, type=float)
    parser.add_argument('--flip_horizontal', action='store_true')
    args = parser.parse_args()

    data_path = os.path.join('..', '..', 'cat-dataset', 'data', 'clean')

    img_size = 224
    exp_name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    exp_name += '_%.5f_%d_%d' % (args.learning_rate, args.units, args.batch_size)
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', '%s.h5' % exp_name)

    pretrained_net = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3),
                                              include_top=False, pooling='max', weights='imagenet')

    path_train = os.path.join(data_path, 'training')
    datagen_train = CatDataGenerator(path=path_train, shuffle=True, batch_size=args.batch_size,
                                     flip_horizontal=args.flip_horizontal)
    path_val = os.path.join(data_path, 'validation')
    datagen_val = CatDataGenerator(path=path_val, shuffle=False, batch_size=args.batch_size,
                                   flip_horizontal=False)
    path_test = os.path.join(data_path, 'test')
    datagen_test = CatDataGenerator(path=path_test, shuffle=False, batch_size=args.batch_size,
                                    flip_horizontal=False)

    output_dim = 4
    print('output_dim', output_dim)

    outp = Dense(args.units, activation='relu')(pretrained_net.output)
    outp = Dense(args.units, activation='relu')(outp)
    outp = Dense(output_dim, activation='linear')(outp)
    model = Model(inputs=pretrained_net.input, outputs=outp)

    model.compile(optimizer=keras.optimizers.Adam(lr=args.learning_rate), loss='mse', metrics=[iou])

    model.summary()

    train_history = model.fit_generator(generator=datagen_train, epochs=args.epochs, shuffle=True,
                                        validation_data=datagen_val,
                                        callbacks=[
                                            TensorBoard(log_dir=os.path.join('logs', exp_name)),
                                            ReduceLROnPlateau(factor=args.ReduceLROnPlateau_factor,
                                                              patience=args.ReduceLROnPlateau_patience, verbose=1,
                                                              monitor='val_iou', mode='max'),
                                            EarlyStopping(patience=(2 * args.ReduceLROnPlateau_patience) + 3, verbose=1,
                                                          monitor='val_iou', mode='max'),
                                            ModelCheckpoint(model_path, verbose=1, save_best_only=True,
                                                            monitor='val_iou', mode='max')
                                        ]
                                        )

    print('Testing...')
    model = load_model(model_path, custom_objects={'iou': iou})
    test_eval = model.evaluate_generator(datagen_test, verbose=1)

    try:
        iter(test_eval)
    except AttributeError:
        test_eval = [test_eval]
    test_metrics = {('test_%s' % k): v for k, v in zip(model.metrics_names, test_eval)}
    print(test_metrics)

    append_hp_result('hpsearch.csv', exp_name, vars(args), train_history.history, test_metrics, 'val_iou', 'max')
