import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from joblib import parallel_backend
from functools import partial
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


class WaterPredictor:
    def __init__(self, path, target_column, independents,
                 test_size=0.2, random_state=SEED):
        self.path = path
        self.target_column = target_column
        self.independents = independents
        self.test_size = test_size
        self.random_state = random_state
        self.test_original = None

        self.scaler = RobustScaler()
        self.x_train = self.x_test = self.y_train = self.y_test = None

    def preprocess(self):

        df = pd.read_csv(self.path)
        df.dropna(inplace=True)  # Remove missing rows

        print("Using independents:", self.independents)

        x = df[self.independents].values
        y = df[self.target_column].values


        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=self.test_size, random_state=self.random_state)

        self.test_original = self.y_test.copy()

        self.y_train = np.log1p(self.y_train)
        self.y_test = np.log1p(self.y_test)

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        print("✅ Preprocessing done — NaNs/Infs in X_train:",
              np.isnan(self.x_train).any(), np.isinf(self.x_train).any())

    def _model_builder(self, hp):
        model = tf.keras.Sequential()
        hp_layers = hp.Choice('layers', [1, 2])
        hp_activation = hp.Choice('activation', ['relu', 'tanh'])
        hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3])

        for i in range(hp_layers):
            units = hp.Int('units', min_value=16, max_value=64, step=16)
            model.add(tf.keras.layers.Dense(units=units, activation=hp_activation))

        model.add(tf.keras.layers.Dense(units=1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
        return model

    def tuner_search(self):
        print("Running tuner search...")
        tuner = kt.Hyperband(self._model_builder,
                             objective='val_loss',
                             max_epochs=25,
                             factor=3,
                             directory='neural_nets/water_predictor',
                             )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001)
        tuner.search(self.x_train, self.y_train, epochs=25, callbacks=[stop_early], validation_split=0.2)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters:", best_hps)

        model = tuner.hypermodel.build(best_hps)
        model.fit(self.x_train, self.y_train, epochs = 250, validation_split=0.2, callbacks=[stop_early])
        model.save('neural_nets/water_predictor/best_model.keras')

        y_pred = model.predict(self.x_test).flatten()
        y_pred = np.expm1(y_pred)
        r2 = r2_score(self.test_original, y_pred)
        mae = mean_absolute_error(self.test_original, y_pred)
        print("R2:", r2)
        print("MAE:", mae)

    def load_best_model(self):
        best_model = tf.keras.models.load_model('saved_models/neural_nets/water_predictor/best_model.keras')
        y_pred = best_model.predict(self.x_test).flatten()
        y_pred = np.expm1(y_pred)
        r2 = r2_score(self.test_original, y_pred)
        mae = mean_absolute_error(self.test_original, y_pred)
        print("R2:", r2)
        print("MAE:", mae)

#Usage
model = WaterPredictor(
    "datasets/water_WEH_%.csv",
    'Mars Odyssey Neutron Spectrometer % WEH',
    [
        'MOLA 128ppd Elevation',
        'OMEGA Est. Lambert Albedo 1080nm',
        'Dayside Thermal Inertia (20 ppd) (Putzig and Mellon 2007)',
        'TES Basalt Abundance - Numeric',
        'Yearly Average Mars Surface Temperature',
        'Latitude (N)',
        'OMEGA Band depth at 2000 nm',
        'Crustal Thickness (km)'
    ]
)
model.preprocess()
#model.tuner_search()
model.load_best_model()
