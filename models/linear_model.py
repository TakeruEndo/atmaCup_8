import os
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import trange
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score

from preprocess import preprocess


class Linear_models:
    def __init__(self, X, y, X_test, output_path, fold_type, n_splits):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.output_path = output_path
        self.fold_type = fold_type
        self.n_splits = n_splits

    def trainer(self):

        y_pred_logr = np.zeros(len(self.X_test))
        scores_logr = []
        y_pred_linr = np.zeros(len(self.X_test))
        scores_linr = []
        y_pred_ridge = np.zeros(len(self.X_test))
        scores_ridge = []
        y_pred_rasso = np.zeros(len(self.X_test))
        scores_rasso = []
        n_splits = 5
        pred_dict = {}
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=71)
        for i, (tr_idx, va_idx) in enumerate(kf.split(self.X)):
            tr_x, va_x = self.X.iloc[tr_idx], self.X.iloc[va_idx]
            tr_y, va_y = self.y.iloc[tr_idx], self.y.iloc[va_idx]

            """ロジスティック回帰
            predict_probで予測値の出力
            """
            print('start_Logistic')
            model_logr = LogisticRegression(C=1.0)
            score, pred = self.train_and_test(
                model_logr, tr_x, tr_y, va_x, va_y, self.X_test, n_splits, i, 'logisticR')
            scores_logr.append(score)
            y_pred_logr += pred / n_splits

            """線形回帰
            """
            print('LinearRegression')
            model_linr = LinearRegression()
            score, pred = self.train_and_test(
                model_linr, tr_x, tr_y, va_x, va_y, self.X_test, n_splits, i, 'linearR')
            scores_linr.append(score)
            y_pred_linr += pred / n_splits

            """リッジ回帰
            alpha（正則化）を調整する
            """
            print('Ridge')
            model_ridge = Ridge(alpha=10)
            score, pred = self.train_and_test(
                model_ridge, tr_x, tr_y, va_x, va_y, self.X_test, n_splits, i, 'RidgeR')
            scores_ridge.append(score)
            y_pred_ridge += pred / n_splits

            """ラッソ回帰
            """
            print('Lasso')
            model_rasso = Lasso(alpha=1).fit(tr_x, tr_y)
            score, pred = self.train_and_test(
                model_rasso, tr_x, tr_y, va_x, va_y, self.X_test, n_splits, i, 'RassoR')
            scores_rasso.append(score)
            y_pred_rasso += pred / n_splits
            print('------------------------')

        pred_dict['logisticR'] = y_pred_logr
        self.log_score(scores_logr, n_splits, 'logisticR')
        pred_dict['linearR'] = y_pred_linr
        self.log_score(scores_linr, n_splits, 'linearR')
        pred_dict['RidgeR'] = y_pred_ridge
        self.log_score(scores_ridge, n_splits, 'RidgeR')
        pred_dict['RassoR'] = y_pred_rasso
        self.log_score(scores_rasso, n_splits, 'RassoR')

        # with open(os.path.join(output_path, 'params.txt'), mode='w') as f:
        #     for k, v in params.items():
        #         f.write("{}: {}".format(k, v))
        #         f.write('\n')
        return pred_dict

    def train_and_test(self, model, tr_x, tr_y, va_x, va_y, X_test, n_splits, fold, model_name):
        model.fit(tr_x, tr_y)
        va_pred = model.predict(va_x)
        score = self.metric(va_y, va_pred)
        
        pred = model.predict(X_test)
        print('{}---------{}/{}_Fold: val_accuracy: {}'.format(model_name,
                                                               fold + 1, n_splits, score))
        logging.info(
            '{}/{}_Fold: val_accuracy: {}'.format(fold + 1, n_splits, score))
        return score, pred

    def log_score(self, score_list, n_splits, model_name):
        y_pred = sum(score_list) / n_splits
        print('{}---------avarage_accuracy: {}'.format(model_name, y_pred))
        logging.info(
            '{}---------avarage_accuracy: {}'.format(model_name, y_pred))

    def metric(self, va_y, pred):
        score = average_precision_score(va_y, pred)
        return score