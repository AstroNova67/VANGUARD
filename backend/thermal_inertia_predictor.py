import pandas as pd
import joblib
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.svm import SVR
from xgboost import XGBRegressor

# df_red_green = pd.read_csv('datasets/raw_data/Thermal_Inertia/region_red-green.csv')
# df_blue = pd.read_csv('datasets/raw_data/Thermal_Inertia/region_blue.csv')
# df_purple = pd.read_csv('datasets/raw_data/Thermal_Inertia/region_purple.csv')
#
#
# print(df_red_green.shape, df_blue.shape, df_purple.shape)
# print(df_red_green.isnull().sum(), df_blue.isnull().sum(), df_purple.isnull().sum())

df = pd.read_csv('datasets/combined_thermal_inertia_region.csv')

df.dropna(inplace=True)
print(df.isnull().sum())


# datasets = [df_red_green, df_blue, df_purple]
# df = pd.concat(datasets, ignore_index = True)
# df.to_csv('datasets/combined_thermal_inertia_region.csv')

class ThermalInertiaPredictor:
    def __init__(self, dataset):

        self.target_transform = 'quantile'

        self.x = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values

        if self.target_transform == 'log':
            self.y = np.log1p(self.y)
        elif self.target_transform == 'quantile':
            self.qt_y = QuantileTransformer(output_distribution='normal')
            self.y = self.qt_y.fit_transform(self.y.reshape(-1, 1)).ravel()

        # split dataset into train and test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=42)

        # Build Models
        self.xgb = XGBRegressor(random_state=42)
        self.reg_svr = SVR(kernel='rbf')
        self.reg_forest = RandomForestRegressor(random_state=42)

        # Feature Scale
        self.sc_x = RobustScaler()
        self.sc_y = RobustScaler()

        self.x_train_scaled = self.sc_x.fit_transform(self.x_train)
        self.x_test_scaled = self.sc_x.transform(self.x_test)
        self.y_train_scaled = self.sc_y.fit_transform(self.y_train.reshape(-1, 1)).ravel()

    def train(self):
        print('training ...')
        self.xgb.fit(self.x_train, self.y_train)
        self.reg_svr.fit(self.x_train_scaled, self.y_train_scaled)
        self.reg_forest.fit(self.x_train, self.y_train)

    def random_search(self, n_iter=100, cv=5):
        print("Running Randomized Search for hyperparameters...")

        # Parameter grids
        param_grid_xgb = {
            'n_estimators': np.arange(50, 500, 50),
            'max_depth': np.arange(2, 12, 1),
            'learning_rate': np.linspace(0.01, 0.3, 10),
            'subsample': np.linspace(0.6, 1.0, 5)
        }

        param_grid_svr = {
            'C': np.logspace(-2, 2, 10),
            'gamma': np.logspace(-3, 1, 10),
            'kernel': ['rbf']
        }

        param_grid_forest = {
            'n_estimators': np.arange(50, 500, 50),
            'max_depth': np.arange(2, 20, 2),
            'min_samples_split': np.arange(2, 10, 1),
            'min_samples_leaf': np.arange(1, 5, 1)
        }

        # Randomized Search for XGBoost
        rs_xgb = RandomizedSearchCV(self.xgb, param_grid_xgb, n_iter=n_iter, cv=cv,
                                    scoring='r2', random_state=42, n_jobs=-1)
        rs_xgb.fit(self.x_train, self.y_train)
        print(f"Best XGB params: {rs_xgb.best_params_}, score: {rs_xgb.best_score_:.3f}")

        # Randomized Search for SVR (scaled data)
        rs_svr = RandomizedSearchCV(self.reg_svr, param_grid_svr, n_iter=n_iter, cv=cv,
                                    scoring='r2', random_state=42, n_jobs=-1)
        rs_svr.fit(self.x_train_scaled, self.y_train_scaled)
        print(f"Best SVR params: {rs_svr.best_params_}, score: {rs_svr.best_score_:.3f}")

        # Randomized Search for Random Forest
        rs_forest = RandomizedSearchCV(self.reg_forest, param_grid_forest, n_iter=n_iter, cv=cv,
                                       scoring='r2', random_state=42, n_jobs=-1)
        rs_forest.fit(self.x_train, self.y_train)
        print(f"Best RF params: {rs_forest.best_params_}, score: {rs_forest.best_score_:.3f}")

        # Update models with best parameters
        self.xgb = rs_xgb.best_estimator_
        self.reg_svr = rs_svr.best_estimator_
        self.reg_forest = rs_forest.best_estimator_

    def plot_histogram(self, bins=10, title="Histogram", xlabel='Thermal Inertia', ylabel='Density', color="skyblue"):
        """
        Plots a histogram for the given data.

        Parameters:
            data (list, array): The data points to plot.
            bins (int): Number of bins for the histogram.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            color (str): Color of the histogram bars.
        """
        plt.figure(figsize=(8, 6))
        plt.hist(self.y, bins=bins, color=color, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def save_model(self):
        """
                Save trained models to disk.
                - SVR and Random Forest with joblib
                - XGBoost with its native save_model
        """
        # Save SVR
        joblib.dump(self.reg_svr, 'saved_models/regression_models/thermal_inertia/svr_model.pkl')
        # Save Random Forest
        joblib.dump(self.reg_forest, "saved_models/regression_models/thermal_inertia/rf_model.pkl")
        # Save XGBoost
        self.xgb.save_model("saved_models/regression_models/thermal_inertia/xgb_model.json")

        print("Models saved: svr_model.pkl, rf_model.pkl, xgb_model.json")

    def load_model(self):
        """
        Load trained models back from disk.
        """
        # Load SVR
        self.reg_svr = joblib.load("saved_models/regression_models/thermal_inertia/svr_model.pkl")
        # Load Random Forest
        self.reg_forest = joblib.load("saved_models/regression_models/thermal_inertia/rf_model.pkl")
        # Load XGBoost
        self.xgb.load_model("saved_models/regression_models/thermal_inertia/xgb_model.json")

        print("Models loaded successfully")

    def predict(self):
        y_pred_xgb = self.xgb.predict(self.x_test)
        y_pred_forest = self.reg_forest.predict(self.x_test)

        if self.target_transform == 'quantile':
            y_test_original = self.qt_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
            y_pred_xgb = self.qt_y.inverse_transform(y_pred_xgb.reshape(-1, 1)).ravel()
            y_pred_forest = self.qt_y.inverse_transform(y_pred_forest.reshape(-1, 1)).ravel()
        elif self.target_transform == 'log':
            y_test_original = np.expm1(self.y_test)
            y_pred_xgb = np.expm1(y_pred_xgb)
            y_pred_forest = np.expm1(y_pred_forest)
        else:
            y_test_original = self.y_test

        score_xgb = r2_score(y_test_original, y_pred_xgb)
        score_forest = r2_score(y_test_original, y_pred_forest)

        print(f'XGBoost Score: {round(score_xgb * 100, 3)}%')
        print(f'Random Forest Score: {round(score_forest * 100, 3)}%')


class ThermalInertiaNeuralNetwork:
    def __init__(self, dataset):
        self.x = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values

        self.qt_y = QuantileTransformer(output_distribution='normal')
        self.qt_x = QuantileTransformer(output_distribution='normal')

        # Split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )

    def preprocessing(self):
        # Feature scale
        self.x_train = self.qt_x.fit_transform(self.x_train)
        self.x_test = self.qt_x.transform(self.x_test)
        self.y_train = self.qt_y.fit_transform(self.y_train.reshape(-1, 1))

    def _model_builder(self, hp):
        network = tf.keras.Sequential()
        hp_layers = hp.Choice('layers', [1, 2])
        hp_activation = hp.Choice('activation', ['relu', 'tanh'])

        # Restrict learning rate
        hp_learning_rate = hp.Float(
            'learning_rate',
            min_value=1e-5,
            max_value=1e-4,
            sampling='log'
        )

        for i in range(hp_layers):
            units = hp.Int('units', min_value=16, max_value=64, step=16)
            network.add(tf.keras.layers.Dense(units=units, activation=hp_activation))

        network.add(tf.keras.layers.Dense(units=1))
        network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
        return network

    def tuner_search(self):
        print("Running tuner search...")
        tuner = kt.Hyperband(self._model_builder,
                             objective='val_loss',
                             max_epochs=25,
                             factor=3,
                             directory='saved_models/neural_nets/thermal_inertia_predictor',
                             )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001)
        tuner.search(self.x_train, self.y_train, epochs=25, callbacks=[stop_early], validation_split=0.2)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters:", best_hps)

        network = tuner.hypermodel.build(best_hps)
        network.fit(self.x_train, self.y_train, epochs=250, validation_split=0.2, callbacks=[stop_early])
        network.save('saved_models/neural_nets/thermal_inertia_predictor/best_model.keras')

        # prediction on transformed
        y_pred_transformed = network.predict(self.x_test)  # No inverse_transform here

        y_pred = self.qt_y.inverse_transform(y_pred_transformed).ravel()

        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        print("R2:", r2)
        print("MAE:", mae)

    def load_best_model(self):
        best_model = tf.keras.models.load_model('saved_models/neural_nets/thermal_inertia_predictor/best_model.keras')
        y_pred_transformed = best_model.predict(self.x_test)  # Already transformed x
        y_pred = self.qt_y.inverse_transform(y_pred_transformed).ravel()
        score_neural = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        print(f'Score (Neural Network Tuned): {round(score_neural * 100, 3)}%')
        print(f'MAE: {round(mae, 3)}')
        # # Original sample
        # x_orig = np.array([[-164.4290009, 0.237795278, 1.2856848239898682, 0.990284503]])
        #
        # # Transform features using the trained QuantileTransformer
        # x_transformed = self.qt_x.transform(x_orig)
        #
        # # Make prediction
        # prediction = best_model.predict(x_transformed)
        #
        # print(self.qt_y.inverse_transform(prediction).ravel())


# network = ThermalInertiaNeuralNetwork(df)
# network.preprocessing()
# network.load_best_model()

model = ThermalInertiaPredictor(df)
# model.plot_histogram()
# model.train()
# model.random_search()
# model.save_model()
model.load_model()
model.predict()

# Metrics

# quantiled transformed
# Score (Neural Network Tuned): 75.092%
# MAE: 30.498

# XGBoost Score: 76.806%
# SVR Score: 72.847%
# Random Forest Score: 76.994%

# Log transform
# XGBoost Score: 76.141%
# SVR Score: 72.64%
# Random Forest Score: 76.663%

# quantiled transformed (untuned)
# XGBoost Score: 77.26%
# Random Forest Score: 77.632%


# XGBoost Score: -3263690.028%
# Random Forest Score: -3304452.767%
# Random Forest cross: [-1.26761662  0.08991035 -0.02908539]%
# XGBoost cross: [-1.23433435  0.13334332 -0.04994441]%
