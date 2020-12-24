import sys
import os
import copy
import logging
from time import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import collections

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import trange
from sklearn.model_selection import KFold, GroupKFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score, roc_auc_score, mean_squared_log_error
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error

import optuna.integration.lightgbm as lgb

from models.utils import save_feature_impotances, save_params


class lightGBM:
    def __init__(self, X, y, X_test, output_path, fold_type, n_splits):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.output_path = output_path
        self.params_ = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'num_round': 10000,
            # 'n_estimators': 10000,
            'learning_rate': 0.0013,
            'max_depth': -1,
            # 'lambda_l2': 1.056,
            'num_leaves': 256,
            'max_bin': 256,
            'bagging_freq': 1,
            'verbose': -1,
            'colsample_bytree': .5,
            'random_state': 72,
            'seed': 1,
        }
        # optuna
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'num_round': 10000,
            # 'n_estimators': 10000,
            'learning_rate': 0.0063,
            'lambda_l1': 0.26691145045524295,
            'lambda_l2': 0.22267006511029266,
            'num_leaves': 183,
            'feature_fraction': 0.5,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_child_samples': 20,
            'seed': 1,
        }
        self.fold_type = fold_type
        self.n_splits = n_splits

    def trainer(self):
        y_pred = np.zeros(len(self.X_test))
        scores = []
        if self.fold_type == 'kfold':
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=71)
            for i, (tr_idx, va_idx) in enumerate(kf.split(self.X)):
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
            self.epoch_log(score, 0, 1)
            y_pred += pred
            final_score = score

        elif self.fold_type == 'skfold':
            # targetでCV
            num_bins = np.int(1 + np.log2(len(self.X)))
            bins = pd.cut(
                self.y,
                bins=num_bins,
                labels=False
            )
            kf = StratifiedKFold(n_splits=self.n_splits,
                                 shuffle=True, random_state=71)
            year = self.X['Year_of_Release'].fillna(-1)
            for i, (tr_idx, va_idx) in enumerate(kf.split(X=self.X, y=bins.values)):
                tr_x, va_x = self.X.iloc[tr_idx], self.X.iloc[va_idx]
                tr_y, va_y = self.y.iloc[tr_idx], self.y.iloc[va_idx]
                # if i == 0:
                #     # optunaでパラメータサーチ
                #     self.parameta_search(tr_x, tr_y, va_x, va_y)
                #     sys.exit()
                score, pred, model = self.train(
                    tr_x, tr_y, va_x, va_y, 'train', self.params)
                scores.append(score)
                self.epoch_log(score, i, self.n_splits)
                y_pred += pred / self.n_splits
            final_score = sum(scores) / self.n_splits
        
        elif self.fold_type == 'group':
            publisher_train = self.X['Publisher_label']
            unique_publisher = self.X['Publisher_label'].unique()
            self.X = self.X.drop('Publisher_label', axis=1)
            self.X_test = self.X_test.drop('Publisher_label', axis=1)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=71)
            for i, (tr_group_idx, va_group_idx) in enumerate(kf.split(unique_publisher)):
                tr_groups, va_groups = unique_publisher[tr_group_idx], unique_publisher[va_group_idx]
                is_tr = publisher_train.isin(tr_groups)
                is_va = publisher_train.isin(va_groups)
                tr_x, va_x = self.X[is_tr], self.X[is_va]
                tr_y, va_y = self.y[is_tr], self.y[is_va]
                score, pred, model = self.train(
                    tr_x, tr_y, va_x, va_y, 'train', self.params)
                scores.append(score)
                self.epoch_log(score, i, self.n_splits)
                y_pred += pred / self.n_splits
            final_score = sum(scores) / self.n_splits

        print('avarage_accuracy: {}'.format(final_score))
        logging.info('avarage_accuracy: {}'.format(final_score))

        feature_importances = model.feature_importances_
        save_feature_impotances(
            self.X.columns, feature_importances, self.output_path)
        save_params(self.params, self.output_path)

        self.adversarial_validation()
        return y_pred

    def train(self, tr_x, tr_y, va_x, va_y, type=None, params=None):
        clf = lgbm.LGBMRegressor(**self.params)

        clf.fit(tr_x, tr_y,
                eval_set=[(va_x, va_y)],
                early_stopping_rounds=200,
                verbose=200)

        pred_i = clf.predict(va_x)
        pred_i = pred_i.clip(0)
        pred_i = np.expm1(pred_i)
        if type != 'adv':
            va_y = np.expm1(va_y)
            score = self.metric(va_y, pred_i)
            pred = clf.predict(self.X_test)
            pred = pred.clip(0)
            pred = np.expm1(pred)
        else:
            score = roc_auc_score(va_y, pred_i)
            pred = None

        return score, pred, clf
    
    def parameta_search(self, tr_x, tr_y, va_x, va_y):
        params = {
            'objective': 'mean_squared_error',
            'metric': 'rmse'
        }
        trains = lgb.Dataset(tr_x, tr_y)
        valids = lgb.Dataset(va_x, va_y)
        best_params, history = {}, []
        model = lgb.train(
            params,
            trains,
            valid_sets=valids,
            verbose_eval=False,
            num_boost_round=1000,
            early_stopping_rounds=5,
            best_params=best_params,
            tuning_history=history)
        print('--------')
        print('best_params: ', best_params)

    def adversarial_validation(self):
        logging.info('-------Start Adversarial validation---------')
        train_df = self.X
        test_df = self.X_test
        whole_df = pd.concat([train_df, test_df], ignore_index=True)
        adv_target_labl = [True] * len(train_df) + [False] * len(test_df)
        adv_target_labl = pd.DataFrame(adv_target_labl)
        adv_params = {
            'num_round': 7000,
            'max_depth': 2,
            # 'lambda_l2': 1.056,
            'num_leaves': 16,
            'max_bin': 16,
        }
        adv_scores = []

        adv_cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=71)
        for i, (tr_idx, va_idx) in enumerate(adv_cv.split(X=whole_df, y=adv_target_labl)):
            if i != 0:
                continue
            tr_x, va_x = whole_df.iloc[tr_idx], whole_df.iloc[va_idx]
            tr_y, va_y = adv_target_labl.iloc[tr_idx], adv_target_labl.iloc[va_idx]

            score, pred, model = self.train(
                tr_x, tr_y, va_x, va_y, 'adv', adv_params)

            adv_scores.append(score)
            self.epoch_log(score, i, self.n_splits)

        final_score = sum(adv_scores) / self.n_splits

        print('adversarial_score: {}'.format(final_score))
        logging.info('adversarial_score: {}'.format(final_score))

        feature_importances = model.feature_importances_
        save_feature_impotances(
            whole_df.columns, feature_importances, self.output_path, 'adv')
        save_params(self.params, self.output_path)

    def epoch_log(self, score, i, n_splits):
        print('{}/{}_Fold: val_Score: {}'.format(i + 1, n_splits, score))
        logging.info(
            '{}/{}_Fold: val_Score: {}'.format(i + 1, n_splits, score))

    def metric(self, va_y, pred):
        score = np.sqrt(mean_squared_log_error(va_y, pred))
        return score

    def pr_auc(self, y_true, y_pred):
        """Custom eval metric"""
        score = average_precision_score(y_true, y_pred)
        return "pr_auc", score, True
