from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import datetime
from keras.applications import mobilenet_v2
from keras.layers import Dense
from keras.models import Model, load_model
import argparse
import keras
import os

from cat_data_generator import CatDataGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--units', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()

    path = os.path.join('..', '..', 'cat-dataset', 'data', 'clean')

    img_size = 224
    exp_name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    exp_name += '_%.5f_%d_%d' % (args.lr, args.units, args.batch_size)
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', '%s.h5' % exp_name)

    pretrained_net = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3),
                                              include_top=False, pooling='max', weights='imagenet')

    path_train = [os.path.join(path, 'CAT_0%d' % i) for i in range(5)]
    datagen_train = CatDataGenerator(paths=path_train, shuffle=True, batch_size=args.batch_size)
    path_val = [os.path.join(path, 'CAT_0%d' % i) for i in (5,)]
    datagen_val = CatDataGenerator(paths=path_val, shuffle=False, batch_size=args.batch_size)
    path_test = [os.path.join(path, 'CAT_0%d' % i) for i in (6,)]
    datagen_test = CatDataGenerator(paths=path_test, shuffle=False, batch_size=args.batch_size)

    output_dim = 4
    print('output_dim', output_dim)

    outp = Dense(args.units, activation='relu')(pretrained_net.output)
    outp = Dense(args.units, activation='relu')(outp)
    outp = Dense(output_dim, activation='linear')(outp)
    model = Model(inputs=pretrained_net.input, outputs=outp)

    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr), loss='mse')

    model.summary()

    train_history = model.fit_generator(generator=datagen_train, epochs=args.epochs, shuffle=True,
                                        validation_data=datagen_val,
                                        callbacks=[
                                            TensorBoard(log_dir=os.path.join('logs', exp_name)),
                                            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
                                            EarlyStopping(patience=8, verbose=1),
                                            ModelCheckpoint(model_path, verbose=1, save_best_only=True)
                                        ]
                                        )

    model = load_model(model_path)
    test_eval = model.evaluate_generator(datagen_test)
    print(test_eval)
