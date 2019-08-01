import os
import argparse
import datetime
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications import mobilenet_v2
from keras.layers import Dense
from keras.models import Model, load_model

from cat_data_generator import CatDataGenerator
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', default=128, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--ReduceLROnPlateau_factor', default=0.5, type=float)
    parser.add_argument('--ReduceLROnPlateau_patience', default=5, type=int)
    parser.add_argument('--loss_fn', default='mse', type=str, choices=['mse', 'iou', 'iou_and_mse_landmarks'])
    parser.add_argument('--iou_and_mse_landmarks_ratio', default=1e-5, type=float)
    parser.add_argument('--include_landmarks', action='store_true')
    parser.add_argument('--flip_horizontal', action='store_true')
    parser.add_argument('--rotate', action='store_true')
    args = parser.parse_args()

    data_path = os.path.join('..', '..', 'cat-dataset', 'data', 'clean')

    exp_name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    exp_name += '_%s' % args.loss_fn
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', '%s.h5' % exp_name)

    path_train = os.path.join(data_path, 'training')
    datagen_train = CatDataGenerator(path=path_train, shuffle=True, batch_size=args.batch_size,
                                     include_landmarks=args.include_landmarks, flip_horizontal=args.flip_horizontal,
                                     rotate=args.rotate)
    test_validation_args = dict(shuffle=False, batch_size=args.batch_size,
                                include_landmarks=args.include_landmarks, flip_horizontal=False,
                                rotate=False)
    path_val = os.path.join(data_path, 'validation')
    datagen_val = CatDataGenerator(path=path_val, **test_validation_args)
    path_test = os.path.join(data_path, 'test')
    datagen_test = CatDataGenerator(path=path_test, **test_validation_args)

    pretrained_net = mobilenet_v2.MobileNetV2(include_top=False, pooling='max')
    outp = Dense(args.units, activation='relu')(pretrained_net.output)
    outp = Dense(args.units, activation='relu')(outp)
    outp = Dense(datagen_train.output_dim, activation='linear')(outp)
    model = Model(inputs=pretrained_net.input, outputs=outp)

    if args.loss_fn in ('iou', 'iou_and_mse_landmarks'):
        # pretrain using mse loss for stability
        model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=[utils.iou])
        model.fit_generator(generator=datagen_train, epochs=1, shuffle=True, steps_per_epoch=50)

    loss_fn = utils.get_loss_fn(args.loss_fn, args.iou_and_mse_landmarks_ratio)
    model.compile(optimizer=keras.optimizers.Adam(lr=args.learning_rate), loss=loss_fn, metrics=[utils.iou, 'mse'])
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
    custom_objects = utils.get_custom_objects(args.loss_fn, loss_fn)
    model = load_model(model_path, custom_objects=custom_objects)
    test_eval = model.evaluate_generator(datagen_test, verbose=1)

    try:
        iter(test_eval)
    except AttributeError:
        test_eval = [test_eval]
    test_metrics = {('test_%s' % k): v for k, v in zip(model.metrics_names, test_eval)}
    print(test_metrics)

    utils.append_hp_result('hpsearch.csv', exp_name, vars(args), train_history.history, test_metrics, 'val_iou', 'max')
