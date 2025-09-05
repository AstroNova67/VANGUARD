import random

import joblib
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
# physical_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Feature columns and target variable
independent_variables = ['Elevation', 'Albedo', 'Day Side Thermal Inertia', 'Slope', 'Roughness 0.6km']
dependent_variable = 'Yearly Average Mars Surface Temperature (C)'

# regions = [
#     'datasets/raw_data/Regions/Region_1.csv',
#     'datasets/raw_data/Regions/Region_2.csv',
#     'datasets/raw_data/Regions/Region_3.csv',
#     'datasets/raw_data/Regions/Region_4.csv',
#     'datasets/raw_data/Regions/Region_5.csv'
# ]
#
# dfs = []
#
# for path in regions:
#     try:
#         df = pd.read_csv(path, encoding='utf-8')
#     except UnicodeDecodeError:
#         df = pd.read_csv(path, encoding='latin1')  # fallback for encoding errors
#
#     df.columns = df.columns.str.strip()  # remove any extra spaces in column names
#     dfs.append(df)
#
# # Combine all into one training DataFrame
# df_train = pd.concat(dfs, ignore_index=True)
#
# # Save combined file
# df_train.to_csv('datasets/combined_regions.csv', index=False)
#
# print("Combined region dataset saved as 'combined_regions.csv'")
df = pd.read_csv('datasets/combined_regions.csv')


# print(df.isnull().sum())
# print(df.shape)


class SurfaceTempRegressor:
    def __init__(self, dataset):
        self.x = dataset[independent_variables].values
        self.y = dataset[dependent_variable].values

        # Models
        # Best parameters(Forest): {'bootstrap': False, 'max_depth': 19, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 123}
        self.reg_forest = RandomForestRegressor(random_state=40, n_estimators=123, bootstrap=False, min_samples_leaf=2,
                                                min_samples_split=6, max_depth=19)
        self.xgb = XGBRegressor(n_estimators=189, learning_rate=np.float64(0.16159252052077827), max_depth=9,
                                subsample=np.float64(0.8050838253488191), random_state=40)
        # Best parameters (XGB): {'learning_rate': np.float64(0.16159252052077827), 'max_depth': 9, 'n_estimators': 189, 'subsample': np.float64(0.8050838253488191)}

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=40)

        print(f"x_train shape: {self.x_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")

    def train_models(self):
        self.reg_forest.fit(self.x_train, self.y_train.ravel())
        self.xgb.fit(self.x_train, self.y_train)

    def graph(self):
        plt.figure(figsize=(8, 6))

        # Fallout 3 green for predictions, blue for perfect line, vegas amber for RF
        fallout_green = '#00FF66'
        blue = 'blue'
        vegas_amber = '#FFBF00'

        # Scatter plot: XGBoost
        plt.scatter(self.y_test, self.y_pred_xgb, color=fallout_green, alpha=0.4, label='XGBoost Predictions')

        # Scatter plot: Random Forest
        plt.scatter(self.y_test, self.y_pred_forest, color=vegas_amber, alpha=0.4, label='Random Forest Predictions')

        # Perfect prediction line
        min_val = min(self.y_test.min(), self.y_pred_xgb.min(), self.y_pred_forest.min())
        max_val = max(self.y_test.max(), self.y_pred_xgb.max(), self.y_pred_forest.max())
        plt.plot([min_val, max_val], [min_val, max_val], color=blue, linestyle='--', label='Perfect Prediction')

        plt.xlabel('True Surface Temperature (°C)')
        plt.ylabel('Predicted Surface Temperature (°C)')
        plt.title('Predicted vs True Surface Temperatures')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def random_search(self):
        # Define the parameter distributions
        param_dist_forest = {
            'n_estimators': randint(100, 301),
            'max_depth': [None] + list(range(5, 21)),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

        param_dist_xgb = {
            'n_estimators': randint(100, 201),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.29),  # Range from 0.01 to 0.3
            'subsample': uniform(0.8, 0.2)  # Range from 0.8 to 1.0
        }

        # Set up the random search
        random_search_forest = RandomizedSearchCV(
            estimator=self.reg_forest,
            param_distributions=param_dist_forest,
            n_iter=50,  # Number of random combinations to try
            cv=3,
            scoring='r2',
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        random_search_xgb = RandomizedSearchCV(
            estimator=self.xgb,
            param_distributions=param_dist_xgb,
            n_iter=50,
            cv=5,
            scoring='r2',
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        # Fit the random search to the data
        random_search_xgb.fit(self.x_train, self.y_train.ravel())
        random_search_forest.fit(self.x_train, self.y_train.ravel())

        # Best parameters
        print('Best parameters (XGB):', random_search_xgb.best_params_)
        best_model_xgb = random_search_xgb.best_estimator_
        print("Best parameters (Forest):", random_search_forest.best_params_)
        best_model_forest = random_search_forest.best_estimator_

        # Evaluate

        y_pred_best_xgb = best_model_xgb.predict(self.x_test)
        y_pred_best_forest = best_model_forest.predict(self.x_test)
        self.score_best_xgb = r2_score(self.y_test, y_pred_best_xgb)
        self.score_best_forest = r2_score(self.y_test, y_pred_best_forest)

    # calculate scores for each model
    def save_model(self):
        """
                Save trained models to disk.
                - Random Forest with joblib
                - XGBoost with its native save_model
        """

        # Save Random Forest
        joblib.dump(self.reg_forest, "saved_models/regression_models/surface_temp/rf_model.pkl")
        # Save XGBoost
        self.xgb.save_model("saved_models/regression_models/surface_temp/xgb_model.json")

        print("Models saved: rf_model.pkl, xgb_model.json")

    def load_model(self):
        """
        Load trained models back from disk.
        """
        # Load Random Forest
        self.reg_forest = joblib.load("saved_models/regression_models/surface_temp/rf_model.pkl")
        # Load XGBoost
        self.xgb.load_model("saved_models/regression_models/surface_temp/xgb_model.json")

        print("Models loaded successfully")

    def print_score(self, z):
        score_labels = [
            "Baseline Random Forest",
            "Baseline XGBoost",
            "Grid Search XGBoost",
            "Grid Search Random Forest"
        ]

        y_pred_forest = self.reg_forest.predict(self.x_test).flatten()
        y_pred_xgb = self.xgb.predict(self.x_test).flatten()

        array = [
            r2_score(self.y_test, y_pred_forest),
            r2_score(self.y_test, y_pred_xgb),
            getattr(self, 'score_best_xgb', None),
            getattr(self, 'score_best_forest', None)
        ]

        score = array[z]
        label = score_labels[z]

        print(f"Score ({label}): {round(score * 100, 3)}%")


class SurfaceTempNetwork:
    def __init__(self, path, target_column, independents,
                 test_size=0.2, random_state=SEED):
        self.path = path
        self.target_column = target_column
        self.independents = independents
        self.test_size = test_size
        self.random_state = random_state
        self.test_original = None
        self.y_min = None  # store min for inverse shift

        self.scaler = RobustScaler()
        self.x_train = self.x_test = self.y_train = self.y_test = None

    def preprocess(self):
        df = pd.read_csv(self.path).dropna()

        # Target shift + log1p
        y = df[self.target_column].values
        self.y_min = y.min()
        y_shifted = y - self.y_min + 1
        y_log = np.log1p(y_shifted)

        df[self.target_column] = y_log
        print("Using independents:", self.independents)

        x = df[self.independents].values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, df[self.target_column].values,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Save original untransformed y_test for metrics
        self.test_original = self.y_test.copy()

        # Scale features
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        print("Preprocessing done — NaNs/Infs in x_train:",
              np.isnan(self.x_train).any(), np.isinf(self.x_train).any())

    def _inverse_transform(self, y_pred_log):
        """Undo log1p + shift."""
        return np.expm1(y_pred_log) + self.y_min - 1

    def _model_builder(self, hp):
        model = tf.keras.Sequential()
        hp_layers = hp.Choice('layers', [1, 2])
        hp_activation = hp.Choice('activation', ['relu', 'tanh'])
        hp_learning_rate = hp.Choice('learning_rate', [1e-3, 1e-4])  # safer

        for i in range(hp_layers):
            units = hp.Int('units', min_value=16, max_value=64, step=16)
            model.add(tf.keras.layers.Dense(units=units, activation=hp_activation))

        model.add(tf.keras.layers.Dense(units=1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

    def tuner_search(self):
        print("Running tuner search...")
        tuner = kt.Hyperband(
            self._model_builder,
            objective='val_loss',
            max_epochs=25,
            factor=3,
            directory='saved_models/neural_nets/surface_temp_pred'
        )

        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, min_delta=0.01
        )

        tuner.search(self.x_train, self.y_train, epochs=25,
                     callbacks=[stop_early], validation_split=0.2)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters:", best_hps)

        model = tuner.hypermodel.build(best_hps)
        model.fit(self.x_train, self.y_train, epochs=250,
                  validation_split=0.2, callbacks=[stop_early])
        model.save('saved_models/neural_nets/surface_temp_pred/best_model.keras')

        y_pred_log = model.predict(self.x_test).flatten()
        y_pred = self._inverse_transform(y_pred_log)
        y_true = self._inverse_transform(self.test_original)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print("R2:", r2)
        print("MAE:", mae)

    def load_best_model(self):
        best_model = tf.keras.models.load_model(
            'saved_models/neural_nets/surface_temp_pred/best_model.keras'
        )
        y_pred_log = best_model.predict(self.x_test).flatten()
        y_pred = self._inverse_transform(y_pred_log)
        y_true = self._inverse_transform(self.test_original)

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"Score (Neural Network): {round(r2 * 100, 3)}%")
        print("MAE:", mae)


# Instantiate and run

model = SurfaceTempRegressor(df)
# model.train_models()
# model.graph()
# model.random_search()
# model.save_model()
model.load_model()
model.print_score(0)
model.print_score(1)
# model.print_score(2)
# model.print_score(3)

model = SurfaceTempNetwork('datasets/combined_regions.csv',
                           'Yearly Average Mars Surface Temperature (C)',
                           ['Elevation', 'Albedo', 'Day Side Thermal Inertia', 'Slope', 'Roughness 0.6km'],
                           )
# model.preprocess()
# # model.tuner_search()
# model.load_best_model()


# Metrics
# Best Random Search (Neural) Parameters: {'n_units': 64, 'n_hidden_layers': 3, 'learning_rate': 0.001, 'epochs': 30, 'batch_size': 32, 'activation': 'tanh'}
# Neural Network
# R2: 0.8621649700697012
# MAE: 4.455943061752762
# Best parameters (XGB): {'learning_rate': np.float64(0.16159252052077827), 'max_depth': 9, 'n_estimators': 189, 'subsample': np.float64(0.8050838253488191)}
# Best parameters (Forest): {'bootstrap': False, 'max_depth': 19, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 123}
# Score (Baseline Random Forest): 92.95%
# Score (Baseline XGBoost): 91.12%
# Score (Grid Search XGBoost): 92.226%
# Score (Grid Search Random Forest): 92.775%
