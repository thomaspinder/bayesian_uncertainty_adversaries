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


class KerasBCNN:
    """
    An implementation of a BCNN in Keras
    """
    def __init__(self, dataset, dropout_p=0.5):
        """
        Initial parameters needed for model initialisation

        :param dataset: A dataset object for which the BCNN should be utilised over
        :param dropout_p: The proportion of dropout to hold within each layer
        """
        self.data=dataset
        self.dropout_p=dropout_p
        self.model = self.build_model(self.dropout_p)
        self.train = None
        self.test = None

    def train_model(self, epochs):
        tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
        callbacks = [EarlyStopping(monitor='val_loss', patience=3),
                     ModelCheckpoint(filepath='src/vision/checkpoint/xray_best.h5', monitor='val_loss',
                                     save_best_only=True),
                     tbCallBack]
        self.model.fit(self.data.train, epochs=epochs, validation_data=self.data.test, callbacks=callbacks)

    def build_model(self, dropout_p=0.5):
        """
        Build a BCNN with dropout layers applied to each of the four layers.

        :param dropout_p: The percentage of dropout to apply to each layer.
        :type dropout_p: float
        """
        inp = Input(shape=self.data.input_shape)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=1)(inp)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
        x = Dropout(dropout_p)(x, training=True)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2)(x)
        x = Dropout(dropout_p)(x, training=True)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_p)(x, training=True)
        x = Dense(self.data.classes, activation='softmax')(x)
        return Model(inp, x, name='lenet-all')

    def mc_pred(self, image, T):
        sto_results = []
        for _ in range(T):
            pred = np.argmax(self.model.predict(image))
            sto_results.append(pred)
        mean_res = np.round(np.mean(sto_results), 0).astype(int)
        var_res = np.var(sto_results)
        return mean_res, var_res

    def construct_adversary(self, image, session, eps):
        wrap = KerasModelWrapper(self.model)
        fgsm = FastGradientMethod(wrap, sess=session)
        fgsm_params = {'eps': eps,
                       'clip_min': 0.,
                       'clip_max': 1.}
        adv_image = fgsm.generate_np(image, **fgsm_params)
        return adv_image

    def mc_dropout(self, stochastic_passes):
        """
        Evaluate a model using Monte-Carlo dropout.

        :param stochastic_passes: The number of stochastic passes to make through the BCNN.
        :type stochastic_passes: int
        :return: Results including prediction, confidence and true label.
        :rtype: DataFrame
        """
        results = []
        for _ in tqdm(range(self.data.test.n), desc='Batching Data'):
            data, label = self.data.test.next()
            sto_results = []
            normal_prediction = np.argmax(self.model.predict(data))
            for _ in range(stochastic_passes):
                pred = np.argmax(self.model.predict(data))
                sto_results.append(pred)
            mean_res = np.round(np.mean(sto_results), 0).astype(int)
            var_res = np.var(sto_results)
            results.append([normal_prediction, mean_res, var_res, int(label)])
        results_df = pd.DataFrame(results, columns=['standard_pred', 'mc_pred', 'mc_conf', 'truth'])
        return results_df

    def mc_adversary(self, stochastic_passes, session, epsilon, obs_counts):
        """
        Evaluate the model against adversarially perturbed images

        :param stochastic_passes: Number of stochastic passes to make when evaluating an observations
        :type stochastic_passes: int
        :param session: TensorFlow session. This is only need for creating adversaries
        :param epsilon: Bound for perturbing images. Value should be within [0,1)
        :type epsilon: float
        :param obs_counts: Number of observations to evaluate. This procedure is timely, so
        :return:
        """
        test_obs = np.min((obs_counts, self.data.test.n))
        results = []
        for _ in tqdm(range(test_obs), desc='Batching Data'):
            data, label = self.data.test.next()
            # Make non-mc prediction
            normal_prediction = np.argmax(self.model.predict(data))

            # Make MC dropout prediction
            mean_res, var_res = self.mc_pred(data, stochastic_passes)

            # Create adversary and re-test
            mc_adversary = self.construct_adversary(data, session, epsilon)
            normal_adv = np.argmax(self.model.predict(mc_adversary))

            # Calculate predictive mean and variance
            mean_adv, var_adv = self.mc_pred(data, stochastic_passes)
            results.append([normal_prediction, mean_res, var_res, normal_adv, mean_adv, var_adv, int(label)])
        results_df = pd.DataFrame(results, columns=['standard_pred', 'mc_pred', 'mc_conf', 'cnn_adv', 'mc_adv',
                                                    'mc_conf', 'truth'])
        results_df['epsilon'] = epsilon
        return results_df

    def compile_model(self, lr, loss, metric):
        self.model.compile(lr, loss=loss, metrics=metric)

    def load_model(self, saved_model):
        self.model = load_model(saved_model)


class KDataset:
    """
    An object to hold and process the dataset being modelled.
    """
    def __int__(self, data_dimension=(224, 224), data_channels=1, data_class_count=2):
        self.data_dims = data_dimension
        self.channels = data_channels
        self.classes = data_class_count
        self.input_shape = data_dimension+(data_channels,)
        self.train = None
        self.test = None

    def load_data(self, data_dir):
        """
        Read in and process the chest x-ray dataset.

        :param data_dir: The directory containing the dataset in terms of data_dir/train and data_dir/test.
        :type data_dir: str
        :return: Train and Test ImageDataGenerator objects
        """
        gen = ImageDataGenerator()
        train_batches = gen.flow_from_directory("{}train".format(data_dir), self.data_dims, color_mode="grayscale",
                                                shuffle=True, seed=1, batch_size=8)
        test_batches = gen.flow_from_directory("{}test".format(data_dir), self.data_dims, shuffle=False,
                                               color_mode="grayscale", batch_size=1, class_mode='binary')
        self.train = train_batches
        self.test = test_batches

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test


if __name__=='__main__':
    args = build_parser()

    # Load Data
    xray_data = KDataset()
    xray_data.load_data('data/chest_xray/')

    # Initialise model
    bcnn = KerasBCNN(xray_data)
    bcnn.build_model(0.5)

    # train, test = get_data()
    if args.train:
        uf.box_print('Training Model')

        # Compile model
        bcnn.compile_model(Adam(lr=0.001), loss='categorical_crossentropy', metric=['accuracy'])
        bcnn.train_model(50)

    else:
        uf.box_print('Loading Weights')
        bcnn.load_model('src/vision/checkpoint/best_model.h5')

    if args.adversary:
        uf.box_print('Crafting {} Adversaries with epsilon = {}'.format(args.observations, args.epsilon))
        # Retrieve the tensorflow session
        sess = backend.get_session()

        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=(None, 224, 224, 1))
        y = tf.placeholder(tf.float32, shape=(None, 2))

        # Collect results
        adv_results = bcnn.mc_adversary(100, sess, args.epsilon, args.observations)
        adv_results.to_csv('results/xray_adv/xray_mc_adv_{}.csv'.format(args.epsilon), index=False)

    if args.monte==2:
        uf.box_print('Testing Monte Carlo Dropout')
        # results = mc_dropout(model, test, 100)
        results = bcnn.mc_dropout(100)
        results.to_csv('results/xray_mc.csv', index=False)

    else:
        pass
