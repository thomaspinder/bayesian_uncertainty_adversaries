import matplotlib.pyplot as plt
import os
import pandas as pd
from time import time
from keras import Model
from keras.models import load_model
from tqdm import tqdm
from keras.layers import MaxPool2D,Dropout,Input, Conv2D, BatchNormalization, Dense, Flatten
from keras.optimizers import Adam
from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import set_random_seed
import tensorflow as tf
import argparse
import src.utils.utility_funcs as uf
from keras.backend.tensorflow_backend import set_session
import cv2
import glob
import numpy as np
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
os.environ['PYTHONHASHSEED'] = "123"
np.random.seed(123)
set_random_seed(123)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', type=bool, default=False)
    parser.add_argument('-m', '--monte', type=int, default=0, choices=[0,1,2],
                        help='(0) - no testing, (1) - test on standard cnn, (2) - test on bcnn.')
    parser.add_argument('-a', '--adversary', type=bool, default=False)
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)
    parser.add_argument('-s', '--small', type=bool, default=False, help='Should the smaller test set be used?')
    args = parser.parse_args()
    return args


def lenet_all(input_shape=(224, 224, 1), num_classes=2):
    inp = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3,3), strides=1)(inp)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2), strides=2)(x)
    x = Dropout(0.5)(x, training=True)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(x)
    # x = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = Dropout(0.5)(x, training=True)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x, training=True)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inp, x, name='lenet-all')


def jpg_to_numpy(directory):
    files = glob.glob('{}/*.jpeg'.format(directory))
    for myFile in files[:1]:
        image = cv2.imread(myFile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape((1, image.shape[0], image.shape[1], 1))
    return image


def get_data(data_dir='data/chest_xray/', small=False):
    # Normalise images to mean +- 1 sd
    gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    fitter = jpg_to_numpy('data/chest_xray/testSmall/normal')
    gen.fit(fitter)
    train_batches = gen.flow_from_directory("{}train".format(data_dir), (224, 224), color_mode="grayscale",
                                            shuffle=True, seed=1, batch_size=8, class_mode='categorical')
    if small:
        test_batches = gen.flow_from_directory("{}testSmall".format(data_dir), (224, 224), shuffle=False,
                                               color_mode="grayscale", batch_size=1, class_mode='categorical')
    else:
        test_batches = gen.flow_from_directory("{}test".format(data_dir), (224, 224), shuffle=False,
                                               color_mode="grayscale", batch_size=1, class_mode='categorical')
    return train_batches, test_batches


def evaluate_model(model, test_data):
    p = model.predict_generator(test_data, verbose=True)
    pre = pd.DataFrame(p)
    pre.columns = ['normal_pred', 'pneumonia_pred']
    print(type(test.filenames))
    pre["filename"] = [x.lower() for x in test.filenames]
    pre["label"] = (pre["filename"].str.contains("pneumonia")).apply(int)
    pre['pre'] = np.where(pre['normal_pred']>pre['pneumonia_pred'], 0, 1)
    pre['result'] = np.where(pre['pre']==pre['label'], 1, 0)
    print('Accuracy: {}%'.format(np.round((np.sum(pre['result'])/pre.shape[0])*100, 2)))
    return pre


def mc_dropout(model, test_data, stochastic_passes):
    results = []
    for _ in tqdm(range(test_data.n), desc='Batching Data'):
        data, label = test_data.next()
        sto_results = []
        normal_prediction = np.argmax(model.predict(data))
        for _ in range(stochastic_passes):
            pred = np.argmax(model.predict(data))
            sto_results.append(pred)
        mean_res = np.round(np.mean(sto_results), 0).astype(int)
        var_res = np.var(sto_results)
        results.append([normal_prediction, mean_res, var_res, int(label)])
    results_df = pd.DataFrame(results, columns=['standard_pred', 'mc_pred', 'mc_conf', 'truth'])
    return results_df


def mc_adversary(model, adversary, test_data, stochastic_passes, epsilon):
    results = []
    for _ in tqdm(range(test_data.n), desc='Batching Data'):
        data, label = test_data.next()
        label = np.argmax(label)

        # Make non-mc prediction
        normal_prediction = np.argmax(model.predict(data))

        # Make MC dropout prediction
        mean_res, var_res = mc_pred(stochastic_passes, model, data)

        # Create adversary and re-test
        adv_image = adversary.generate(data, eps=epsilon)
        if _ ==0:
            plt.imshow(adv_image.squeeze(), cmap='gray')
            plt.savefig('results/plots/fgsm_imgs/xray_adversary_{}.png'.format(epsilon))
        normal_adv = np.argmax(model.predict(adv_image))
        mean_adv, var_adv = mc_pred(stochastic_passes, model, adv_image)
        results.append([normal_prediction, mean_res, var_res, normal_adv, mean_adv, var_adv, int(label)])
    results_df = pd.DataFrame(results, columns=['standard_pred', 'mc_pred', 'mc_conf', 'cnn_adv', 'mc_adv',
                                                'mc_conf', 'truth'])
    try:
        acc_val = np.sum(results_df.mc_adv.values == results_df.truth.values)
        print('Accuracy: {}'.format(acc_val))
    except:
        print('Tom - Error in accuracy calc')
    results_df['epsilon'] = epsilon
    return results_df


def mc_pred(T, model, data):
    sto_results = []
    for _ in range(T):
        pred = np.argmax(model.predict(data))
        sto_results.append(pred)
    mean_res = np.round(np.mean(sto_results), 0).astype(int)
    var_res = np.var(sto_results)
    return mean_res, var_res


if __name__=='__main__':
    args = build_parser()

    # Load Data
    train, test = get_data(small=args.small)
    test_sample, _ = train.next()
    data_min, data_max = np.amax(test_sample), np.amin(test_sample)
    if args.train:
        uf.box_print('Training Model')

        # Compile model
        model = lenet_all()
        model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

        # Define Callbacks
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                     ModelCheckpoint(filepath='src/vision/checkpoint/xray_best-unnorm.h5', monitor='val_loss', save_best_only=True),
                     tbCallBack]

        # Train model
        start = time()
        model.fit_generator(train, epochs=50, validation_data=test, callbacks=callbacks)
        end = time()-start
        print('Training Time: {}'.format(end))
    else:
        weights_name = 'xray_best-unnorm.h5'
        uf.box_print('Loading Weights From {}'.format(weights_name))
        model = load_model('src/vision/checkpoint/{}'.format(weights_name))

    if args.adversary:
        uf.box_print('Crafting {} Adversaries with epsilon = {}'.format(test.n, args.epsilon))

        # Build adversary constructor
        classifier = KerasClassifier((-3, 3), model=model)
        adv_crafter = FastGradientMethod(classifier)
        adv_results = mc_adversary(model, adv_crafter, test, 100, args.epsilon)
        adv_results.to_csv('results/xray_adv/xray_mc_adv_{}.csv'.format(args.epsilon), index=False)

    if args.monte==2:
        uf.box_print('Testing Monte Carlo Dropout')
        results = mc_dropout(model, test, 100)
        results.to_csv('results/xray_mc.csv', index=False)

    elif args.monte==1:
        uf.box_print('Testing Standard Evaluation')
        # Evaluate model
        prediction_df = evaluate_model(model, test)
        prediction_df.to_csv('results/xray_cnn.csv', index=False)

    else:
        pass
