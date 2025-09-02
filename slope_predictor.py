import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

df_white = pd.read_csv('datasets/raw_data/Slope_Prediction_Dataset/Region_White.csv')
df_gray = pd.read_csv('datasets/raw_data/Slope_Prediction_Dataset/Region_Gray.csv')
df_black = pd.read_csv('datasets/raw_data/Slope_Prediction_Dataset/Region_Black.csv')

# print(df_white.isnull().sum())
# print(df_white.shape)
# print(df_gray.isnull().sum())
# print(df_gray.shape)
# print(df_black.isnull().sum())
# print(df_black.shape)

# df_combined = pd.concat([df_white, df_gray, df_black], ignore_index = True)
#
# df_combined.to_csv('datasets/combined_slope_regions.csv', index = False)

# df = pd.read_csv('datasets/combined_slope_regions.csv')
# 163,690 rows

# Rolling window stats
# df['Elevation_rolling_mean'] = df['Elevation'].rolling(window=5, center=True).mean()
# df['Elevation_rolling_max_diff'] = (
#         df['Elevation'].rolling(window=5, center=True).max() -
#         df['Elevation'].rolling(window=5, center=True).min()
# )
#
# # First and second differences
# df['Elevation_diff'] = df['Elevation'].diff()
# df['Elevation_diff_abs'] = df['Elevation_diff'].abs()
#
# df['Elevation_diff2'] = df['Elevation_diff'].diff()
# df['Elevation_diff2_abs'] = df['Elevation_diff2'].abs()
#
# # Drop raw and intermediate columns
# df.drop(columns=['Elevation', 'Elevation_diff', 'Elevation_diff2'], inplace=True)
# df.to_csv('datasets/combined_slope_regions.csv', index=False)

df = pd.read_csv('datasets/combined_slope_regions.csv')

df['Slope_Log'] = np.log1p(df['Slope (degrees)'])
independent_variables = ['Albedo', 'Day Side Thermal Inertia', 'Roughness', 'OMEGA Ferric/Dust 860nm ratio',
                         'Elevation_rolling_mean', 'Elevation_rolling_max_diff', 'Elevation_diff_abs',
                         'Elevation_diff2_abs']
dependent_variable = ['Slope_Log']


class SlopeNeuralNetwork:
    def __init__(self, datasets):
        self.x = datasets[independent_variables].values
        self.y = datasets[dependent_variable].values
        self.test_original = None

    def preprocessing(self, test_size=0.2, random_state=42):
        # Split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state
        )
        self.test_original = self.y_test.copy()
        # Feature scale inputs
        self.sc = RobustScaler()
        self.x_train = self.sc.fit_transform(self.x_train)
        self.x_test = self.sc.transform(self.x_test)

    def _model_builder(self, hp):
        model = tf.keras.Sequential()
        hp_layers = hp.Choice('layers', [1, 2])
        hp_activation = hp.Choice('activation', ['relu', 'tanh'])
        hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3])

        for i in range(hp_layers):
            units = hp.Int('units', min_value=16, max_value=64, step=16)
            model.add(tf.keras.layers.Dense(units=units, activation=hp_activation))

        model.add(tf.keras.layers.Dense(units=1))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])
        return model

    def tuner_search(self):
        print("Running tuner search...")
        tuner = kt.Hyperband(self._model_builder,
                             objective='val_loss',
                             max_epochs=25,
                             factor=3,
                             directory='neural_nets/slope_pred',
                             )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001)
        tuner.search(self.x_train, self.y_train, epochs=25, callbacks=[stop_early], validation_split=0.2)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters:", best_hps)

        model = tuner.hypermodel.build(best_hps)
        model.fit(self.x_train, self.y_train, epochs=250, validation_split=0.2, callbacks=[stop_early])
        model.save('neural_nets/slope_pred/best_model.keras')

        y_pred = model.predict(self.x_test).flatten()
        r2 = r2_score(self.test_original, y_pred)
        mae = mean_absolute_error(self.test_original, y_pred)
        print("R2:", r2)
        print("MAE:", mae)

    def load_best_model(self):
        best_model = tf.keras.models.load_model('saved_models/neural_nets/slope_pred/best_model.keras')
        y_pred = best_model.predict(self.x_test).flatten()
        score_neural = r2_score(self.test_original, y_pred)
        mae = mean_absolute_error(self.test_original, y_pred)
        print(f'Score (Neural Network Tuned): {round(score_neural * 100, 3)}%')
        print(f'MAE: {round(mae, 3)}')


class SlopeRegressor:
    def __init__(self, datasets):
        self.x = datasets[independent_variables].values
        self.y = datasets[dependent_variable[0]].values
        self.counter = 0

    def preprocessing(self):
        # split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=42)

        # Feature Scale (not required here)
        # self.sc = StandardScaler()
        # self.x_train = self.sc.fit_transform(self.x_train)
        # self.x_test = self.sc.transform(self.x_test)

    def train(self):
        self.reg_forest = RandomForestRegressor(n_estimators=200, random_state=42)
        # Best parameters (Forest): {'bootstrap': False, 'max_depth': 19, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 123}

        self.xgb = XGBRegressor(n_estimators=195, learning_rate=0.04128851382805829, max_depth=9,
                                subsample=0.9272820822527561, random_state=42)
        # Best parameters (XGB): {'learning_rate': 0.04128851382805829, 'max_depth': 9, 'n_estimators': 195, 'subsample': 0.9272820822527561}

        self.reg_forest.fit(self.x_train, self.y_train)
        self.xgb.fit(self.x_train, self.y_train)

        # used to check feature importance
        # importances = self.reg_forest.feature_importances_
        # features = independent_variables
        # sorted_idx = np.argsort(importances)[::-1]
        #
        # print("\nFeature Importances (Random Forest):")
        # for i in sorted_idx:
        #     print(f"{features[i]}: {importances[i]:.4f}")

    def random_search(self):
        self.counter = 1
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

    def graph(self):
        plt.hist(df['Slope_Log'], bins=50)
        plt.title('Slope Value Distribution')
        plt.xlabel('Slope_Log')
        plt.ylabel('Count')
        plt.show()

    def predict(self, index):
        if self.counter == 0:
            array = [self.reg_forest, self.xgb]
            y_pred = array[index].predict(self.x_test)
            score = r2_score(self.y_test, y_pred)
            if index == 0:
                print(f'Score (Random Forest): {round(score * 100, 3)}%')
            else:
                print(f'Score (XGBoost): {round(score * 100, 3)}%')
        else:
            array_best = [self.score_best_xgb, self.score_best_forest]
            if index == 0:
                print(f'Score (Random Forest Tuned): {round(array_best[index] * 100, 3)}%')
            else:
                print(f'Score (XGBoost Tuned): {round(array_best[index] * 100, 3)}%')


model = SlopeNeuralNetwork(df)
model.preprocessing()
# model.tuner_search()
model.load_best_model()
# model.predict()

# regressor = SlopeRegressor(df)
# regressor.preprocessing()
# regressor.train()
# regressor.graph()
# # regressor.random_search()
# regressor.predict(0)
# regressor.predict(1)

# Metrics
# Score (Neural Network Tuned): 80.517%
# MAE: 0.221
# Score (Random Forest): 81.665%
# Score (XGBoost): 82.097%
# Best parameters (XGB): {'learning_rate': 0.04128851382805829, 'max_depth': 9, 'n_estimators': 195, 'subsample': 0.9272820822527561}
# Best parameters (Forest): {'bootstrap': False, 'max_depth': 19, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 123}
