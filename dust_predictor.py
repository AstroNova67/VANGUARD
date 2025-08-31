import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
# import matplotlib.pyplot as plt
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from joblib import parallel_backend
from functools import partial
import tensorflow as tf
import keras_tuner as kt


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
print("GPUs:", tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


class DustPredictor:
    def __init__(self, path, target_column, independents,
                 transform='log', test_size=0.2, random_state=SEED):
        """
        transform: 'log' or 'quantile'
        """
        self.path = path
        self.target_column = target_column
        self.independents = independents
        self.test_size = test_size
        self.random_state = random_state
        self.test_original = None
        self.transform = transform

        self.scaler = RobustScaler()
        self.q_transformer = QuantileTransformer(output_distribution='normal') if transform=='quantile' else None

        self.x_train = self.x_test = self.y_train = self.y_test = None

    def preprocess(self):
        df = pd.read_csv(self.path, header=0)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        df.dropna(inplace=True)

        df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')
        df[self.independents] = df[self.independents].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=[self.target_column] + self.independents)

        print("Using independents:", self.independents)

        x = df[self.independents].values
        y = df[self.target_column].values.reshape(-1, 1)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )

        self.test_original = self.y_test.copy()  # keep untransformed

        # Apply chosen transform
        if self.transform == 'log':
            self.y_train = np.log1p(self.y_train)
            self.y_test = np.log1p(self.y_test)
        elif self.transform == 'quantile':
            self.y_train = self.q_transformer.fit_transform(self.y_train)
            self.y_test = self.q_transformer.transform(self.y_test)

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        print("Preprocessing done — NaNs/Infs in X_train:",
              np.isnan(self.x_train).any(), np.isinf(self.x_train).any())

    def _inverse_transform(self, y_pred):
        if self.transform == 'log':
            return np.expm1(y_pred)
        elif self.transform == 'quantile':
            return self.q_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    def _model_builder(self, hp):
        model = tf.keras.Sequential()

        # Hyperparameters
        hp_layers = hp.Int('layers', min_value=1, max_value=4, step=1)
        hp_units = hp.Int('units', min_value=32, max_value=256, step=32)
        hp_activation = hp.Choice('activation', ['relu', 'tanh', 'elu', 'selu'])
        hp_learning_rate = hp.Choice('learning_rate', [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
        hp_dropout = hp.Float('dropout_rate', min_value=0.0, max_value=0.4, step=0.1)
        hp_batch_norm = hp.Boolean('batch_norm')

        for i in range(hp_layers):
            model.add(tf.keras.layers.Dense(units=hp_units, activation=hp_activation,
                                            kernel_initializer='he_normal'))
            if hp_batch_norm:
                model.add(tf.keras.layers.BatchNormalization())
            if hp_dropout > 0:
                model.add(tf.keras.layers.Dropout(rate=hp_dropout))

        # Output layer
        model.add(tf.keras.layers.Dense(1))
        if self.transform == 'quantile':
            # Optionally add linear activation explicitly
            model.add(tf.keras.layers.Activation('linear'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

    def tuner_search(self):
        print("Running tuner search...")
        tuner = kt.Hyperband(self._model_builder,
                             objective='val_loss',
                             max_epochs=25,
                             factor=3,
                             directory='saved_models/neural_nets/dust_predictor',
                             overwrite=True)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, min_delta=1e-8)
        tuner.search(self.x_train, self.y_train, epochs=25, callbacks=[stop_early],
                     validation_split=0.1, batch_size=512)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters:", best_hps)

        model = tuner.hypermodel.build(best_hps)
        model.fit(self.x_train, self.y_train, epochs=125, validation_split=0.2,
                  callbacks=[stop_early], batch_size=512)
        model.save('saved_models/neural_nets/dust_predictor/best_model.keras')
        model.save('saved_models/neural_nets/dust_predictor/best_model.h5')

        y_pred = model.predict(self.x_test).flatten()
        y_pred = self._inverse_transform(y_pred)

        r2 = r2_score(self.test_original, y_pred)
        mae = mean_absolute_error(self.test_original, y_pred)
        print("R2:", r2)
        print("MAE:", mae)

    def load_best_model(self):
        model_path = 'saved_models/neural_nets/dust_predictor/best_model.keras'
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}. Please run `tuner_search()` first.")
            return

        print("Loading model from:", model_path)
        best_model = tf.keras.models.load_model(model_path)

        y_pred = best_model.predict(self.x_test).flatten()
        y_pred = self._inverse_transform(y_pred)

        r2 = r2_score(self.test_original, y_pred)
        mae = mean_absolute_error(self.test_original, y_pred)
        print("R2:", r2)
        print("MAE:", mae)


# Usage
model = DustPredictor(
    "datasets/dust_small.csv",
    'OMEGA Ferric/Dust 860nm ratio',
    [
        "Elevation",
        "Slope",
        "Yearly Average Mars Surface Temperature",
        "Dayside Thermal Inertia",
        "MOLA 128ppd Aspect",
        "Albedo"
    ]
)

transform = 'quantile'   # choose 'log' or 'quantile'

model.preprocess()
# model.tuner_search()
model.load_best_model()

# Metrics
# R2: 0.854063568466665
# MAE: 0.006056298268290478