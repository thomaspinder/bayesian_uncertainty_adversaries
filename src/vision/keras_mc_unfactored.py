import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from keras import Model
from keras.models import load_model
from tqdm import tqdm
from keras.layers import MaxPool2D,Dense,Dropout,Input, Conv2D, BatchNormalization
from keras.optimizers import Adam
from keras.layers import Dense, Flatten
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import set_random_seed
import tensorflow as tf
from keras import backend
import argparse
import src.utils.utility_funcs as uf
from keras.backend.tensorflow_backend import set_session
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
    parser.add_argument('-o', '--observations', type=int, default=300, help='Number of observations to be used in testing.')
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

def get_data(data_dir='data/chest_xray/'):
    gen = ImageDataGenerator()
    train_batches = gen.flow_from_directory("{}train".format(data_dir), (224, 224), color_mode="grayscale", shuffle=True,
                                            seed=1, batch_size=8)
    test_batches = gen.flow_from_directory("{}test".format(data_dir), (224, 224), shuffle=False, color_mode="grayscale",
                                           batch_size=1, class_mode='binary')
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


def mc_adversary(model, test_data, stochastic_passes, session, epsilon, obs_counts):
    test_obs = np.min((obs_counts, test_data.n))
    results = []
    for _ in tqdm(range(test_obs), desc='Batching Data'):
        data, label = test_data.next()
        # Make non-mc prediction
        normal_prediction = np.argmax(model.predict(data))

        # Make MC dropout prediction
        mean_res, var_res = mc_pred(stochastic_passes, model, data)

        # Create adversary and re-test
        mc_adversary = construct_adversary(model, data, session, epsilon)
        normal_adv = np.argmax(model.predict(mc_adversary))
        mean_adv, var_adv = mc_pred(stochastic_passes, model, mc_adversary)
        results.append([normal_prediction, mean_res, var_res, normal_adv, mean_adv, var_adv, int(label)])
    results_df = pd.DataFrame(results, columns=['standard_pred', 'mc_pred', 'mc_conf', 'cnn_adv', 'mc_adv',
                                                'mc_conf', 'truth'])
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


def construct_adversary(model, image, session, eps):
    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=session)
    fgsm_params = {'eps': eps,
                   'clip_min': 0.,
                   'clip_max': 1.}
    adv_image = fgsm.generate_np(image, **fgsm_params)
    return adv_image


if __name__=='__main__':
    args = build_parser()

    # Load Data
    train, test = get_data()
    if args.train:
        uf.box_print('Training Model')

        # Compile model
        model = lenet_all()
        model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

        # Define Callbacks
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                     ModelCheckpoint(filepath='src/vision/checkpoint/xray_best.h5', monitor='val_loss', save_best_only=True),
                     tbCallBack]

        # Train model
        model.fit_generator(train,epochs=50, validation_data=test, callbacks=callbacks)

    else:
        uf.box_print('Loading Weights')
        model = load_model('src/vision/checkpoint/best_model.h5')

    if args.adversary:
        uf.box_print('Crafting {} Adversaries with epsilon = {}'.format(args.observations, args.epsilon))
        # Retrieve the tensorflow session
        sess = backend.get_session()

        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=(None, 224, 224, 1))
        y = tf.placeholder(tf.float32, shape=(None, 2))

        adv_results = mc_adversary(model, test, 100, sess, args.epsilon, args.observations)
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
