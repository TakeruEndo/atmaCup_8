import sys
import os
import copy
import logging
from time import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import trange
from sklearn.model_selection import KFold, GroupKFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score, mean_squared_log_error
from catboost import CatBoost, Pool, CatBoostClassifier, CatBoostRegressor

from models.utils import save_feature_impotances, save_params


class Catboost:
    def __init__(self, X, y, X_test, output_path, fold_type, n_splits):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.output_path = output_path
        self.params = {
            'loss_function': 'RMSE',
            'iterations': 50000,
            # 'depth': 10,
            'colsample_bylevel': 0.5,
            'early_stopping_rounds': 300,
            'l2_leaf_reg': 18,
            'random_seed': 42,
            'use_best_model': True
        }
        self.fold_type = fold_type
        self.n_splits = n_splits

    def trainer(self):
        y_pred = np.zeros(len(self.X_test))
        scores = []
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=71)
        if self.fold_type == 'kfold':
            for i, (tr_idx, va_idx) in enumerate(kf.split(self.X)):
                print('----------')
                print(f'start_{i}_fold_traingn')
                tr_x, va_x = self.X.iloc[tr_idx], self.X.iloc[va_idx]
                tr_y, va_y = self.y.iloc[tr_idx], self.y.iloc[va_idx]

                score, pred, model = self.train(tr_x, tr_y, va_x, va_y)
                scores.append(score)
                self.epoch_log(score, i, self.n_splits)

                y_pred += pred / self.n_splits

            final_score = sum(scores) / self.n_splits

        elif self.fold_type == 'oof':
            tr_x, va_x, tr_y, va_y = train_test_split(
                self.X, self.y, test_size=0.2)

            score, pred, model = self.train(tr_x, tr_y, va_x, va_y)
            scores.append(score)
            self.epoch_log(score, 0, 1)
            y_pred += pred
            final_score = score

        elif self.fold_type == 'skfold':
            num_bins = np.int(1 + np.log2(len(self.X)))
            bins = pd.cut(
                self.y,
                bins=num_bins,
                labels=False
            )
            kf = StratifiedKFold(n_splits=self.n_splits,
                                 shuffle=True, random_state=71)
            for i, (tr_idx, va_idx) in enumerate(kf.split(X=self.X, y=bins.values)):
                # if i != 0:
                #     continue
                tr_x, va_x = self.X.iloc[tr_idx], self.X.iloc[va_idx]
                tr_y, va_y = self.y.iloc[tr_idx], self.y.iloc[va_idx]

                score, pred, model = self.train(tr_x, tr_y, va_x, va_y)
                scores.append(score)
                self.epoch_log(score, i, self.n_splits)

                y_pred += pred / self.n_splits

            final_score = sum(scores) / self.n_splits

        print('avarage_Score: {}'.format(final_score))
        logging.info('avarage_Score: {}'.format(final_score))

        feature_importances = model.get_feature_importance()
        save_feature_impotances(
            self.X.columns, feature_importances, self.output_path)
        save_params(self.params, self.output_path)
        return y_pred

    def train(self, tr_x, tr_y, va_x, va_y):
        train_pool = Pool(tr_x, label=tr_y)
        test_pool = Pool(va_x, label=va_y)

        model = CatBoostRegressor(**self.params)

        model.fit(train_pool,
                  eval_set=[test_pool],
                  verbose=False,
                  use_best_model=True)

        y_val_pred = model.predict(va_x)
        y_val_pred = np.expm1(y_val_pred)
        y_val_pred = y_val_pred.clip(0)
        va_y = np.expm1(va_y)
        va_y = va_y.clip(0) 
        score = self.metric(va_y, y_val_pred)
        pred = model.predict(self.X_test)
        pred = np.expm1(pred)

        return score, pred, model

    def epoch_log(self, score, i, n_splits):
        print('{}/{}_Fold: val_accuracy: {}'.format(i + 1, n_splits, score))
        logging.info(
            '{}/{}_Fold: val_accuracy: {}'.format(i + 1, n_splits, score))

    def metric(self, va_y, pred):
        score = np.sqrt(mean_squared_log_error(va_y, pred))
        return score
