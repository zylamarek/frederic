import os
import argparse
import datetime
import keras
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.applications import mobilenet_v2
from keras.layers import Dense, Flatten
from keras.models import Model, load_model
from PIL import Image

import context
from frederic.cat_data_generator import CatDataGenerator
import frederic.utils.general

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default=os.path.join('..', '..', 'cat-dataset', 'data', 'clean'))
    parser.add_argument('--output_type', default='bbox', type=str, choices=['bbox', 'landmarks'])
    parser.add_argument('--hpsearch_file', default='hpsearch.csv', type=str)
    parser.add_argument('--units', default=128, type=int)
    parser.add_argument('--pooling', default='max', type=str, choices=['max', 'avg', 'None'])
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--ReduceLROnPlateau_factor', default=0.6, type=float)
    parser.add_argument('--ReduceLROnPlateau_patience', default=8, type=int)
    parser.add_argument('--loss_fn', default='mse', type=str, choices=['mse', 'iou', 'iou_and_mse_landmarks'])
    parser.add_argument('--iou_and_mse_landmarks_ratio', default=1e-5, type=float)
    parser.add_argument('--include_landmarks', action='store_true')
    parser.add_argument('--flip_horizontal', action='store_true')
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--rotate_90', action='store_true')
    parser.add_argument('--rotate_n', default=0, type=int)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--crop_scale_balanced_black', action='store_true')
    parser.add_argument('--crop_scale_balanced', action='store_true')
    args = parser.parse_args()
    if args.pooling == 'None':
        args.pooling = None

    exp_name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    exp_name += '_%s' % args.loss_fn
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', '%s.h5' % exp_name)

    path_train = os.path.join(args.data_path, 'training')
    if args.output_type == 'bbox':
        path_val = os.path.join(args.data_path, 'validation')
        path_test = os.path.join(args.data_path, 'test')
    else:
        path_val = os.path.join(args.data_path, 'landmarks_validation')
        path_test = os.path.join(args.data_path, 'landmarks_test')

    datagen_train = CatDataGenerator(path=path_train,
                                     output_type=args.output_type,
                                     include_landmarks=args.include_landmarks,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     flip_horizontal=args.flip_horizontal,
                                     rotate=args.rotate,
                                     rotate_90=args.rotate_90,
                                     rotate_n=args.rotate_n,
                                     crop=args.crop,
                                     crop_scale_balanced_black=args.crop_scale_balanced_black,
                                     crop_scale_balanced=args.crop_scale_balanced,
                                     sampling_method_resize='random')
    test_validation_args = dict(output_type=args.output_type,
                                include_landmarks=args.include_landmarks,
                                batch_size=args.batch_size,
                                shuffle=False,
                                flip_horizontal=False,
                                rotate=False,
                                rotate_90=False,
                                rotate_n=0,
                                crop=False,
                                crop_scale_balanced_black=False,
                                crop_scale_balanced=False,
                                sampling_method_resize=Image.LANCZOS)
    datagen_val = CatDataGenerator(path=path_val, **test_validation_args)
    datagen_test = CatDataGenerator(path=path_test, **test_validation_args)

    pretrained_net = mobilenet_v2.MobileNetV2(input_shape=frederic.utils.general.IMG_SHAPE, include_top=False,
                                              pooling=args.pooling)
    outp = pretrained_net.output
    if args.pooling is None:
        outp = Flatten()(outp)
    outp = Dense(args.units, activation='relu')(outp)
    outp = Dense(args.units, activation='relu')(outp)
    outp = Dense(datagen_train.output_dim, activation='linear')(outp)
    model = Model(inputs=pretrained_net.input, outputs=outp)

    if args.loss_fn in ('iou', 'iou_and_mse_landmarks') and args.output_type == 'bbox':
        # Pretrain using mse loss for stability. IOU based losses easily explode in the beginning of the training.
        model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=[frederic.utils.general.iou])
        model.fit_generator(generator=datagen_train, epochs=1, shuffle=True, steps_per_epoch=50, workers=3)

    if args.output_type == 'bbox':
        metrics = [frederic.utils.general.iou, 'mse']
        monitor, mode = 'val_iou', 'max'
    else:
        metrics = []
        monitor, mode = 'val_loss', 'min'
    loss_fn = frederic.utils.general.get_loss_fn(args.output_type, args.loss_fn, args.iou_and_mse_landmarks_ratio)

    model.compile(optimizer=keras.optimizers.Adam(lr=args.learning_rate), loss=loss_fn, metrics=metrics)
    model.summary()

    train_history = model.fit_generator(generator=datagen_train, epochs=args.epochs, shuffle=True,
                                        validation_data=datagen_val, workers=3,
                                        callbacks=[
                                            TensorBoard(log_dir=os.path.join('logs', exp_name)),
                                            ReduceLROnPlateau(factor=args.ReduceLROnPlateau_factor,
                                                              patience=args.ReduceLROnPlateau_patience, verbose=1,
                                                              monitor=monitor, mode=mode),
                                            EarlyStopping(patience=(2 * args.ReduceLROnPlateau_patience) + 3, verbose=1,
                                                          monitor=monitor, mode=mode),
                                            ModelCheckpoint(model_path, verbose=1, save_best_only=True,
                                                            monitor=monitor, mode=mode)
                                        ]
                                        )

    print('Testing...')
    custom_objects = frederic.utils.general.get_custom_objects(args.loss_fn, loss_fn)
    model = load_model(model_path, custom_objects=custom_objects)
    test_eval = model.evaluate_generator(datagen_test, verbose=1)

    try:
        iter(test_eval)
    except TypeError:
        test_eval = [test_eval]
    test_metrics = {('test_%s' % k): v for k, v in zip(model.metrics_names, test_eval)}
    print(test_metrics)

    frederic.utils.general.append_hp_result(path=args.hpsearch_file, exp_name=exp_name, args=vars(args),
                                            history=train_history.history, test_metrics=test_metrics, monitor=monitor,
                                            mode=mode)
