import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import KFold, GroupKFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score

from models.utils import save_feature_impotances, save_params


class XGBoost:
    def __init__(self, X, y, X_test, output_path, fold_type, n_splits):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.output_path = output_path
        self.params = {
            'objective': 'binary:logistic',
            # 'eval_metric': 'logloss',
            'max_depth': 10,
            'subsample': 0.5,
            'learning': 0.0013,
            'random_state': 42
        }
        self.fold_type = fold_type
        self.n_splits = n_splits

    def trainer(self):
        y_pred = np.zeros(len(self.X_test))
        scores = []
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=71)
        if self.fold_type == 'kfold':
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
            scores.append(score)
            self.epoch_log(score, 0, 1)
            y_pred += pred
            final_score = score

        elif self.fold_type == 'skfold':
            kf = StratifiedKFold(n_splits=self.n_splits,
                                 shuffle=True, random_state=71)
            for i, (tr_idx, va_idx) in enumerate(kf.split(X=self.X, y=self.y)):
                tr_x, va_x = self.X.iloc[tr_idx], self.X.iloc[va_idx]
                tr_y, va_y = self.y.iloc[tr_idx], self.y.iloc[va_idx]

                score, pred, model = self.train(tr_x, tr_y, va_x, va_y)
                scores.append(score)
                self.epoch_log(score, i, self.n_splits)

                y_pred += pred / self.n_splits

            final_score = sum(scores) / self.n_splits

        print('avarage_accuracy: {}'.format(final_score))
        logging.info('avarage_accuracy: {}'.format(final_score))

        feature_importances = model.get_score(importance_type='total_gain')
        save_feature_impotances(
            feature_importances.keys(), feature_importances.values(), self.output_path)
        save_params(self.params, self.output_path)
        return y_pred

    def train(self, tr_x, tr_y, va_x, va_y):
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        dvalid = xgb.DMatrix(va_x, label=va_y)
        dtest = xgb.DMatrix(self.X_test)

        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        model = xgb.train(
            self.params,
            dtrain,
            evals=watchlist,
            feval=self.pr_auc,
            early_stopping_rounds=200,
            verbose_eval=200
        )

        y_val_pred = model.predict(dvalid)
        score = self.metric(va_y, y_val_pred)

        pred = model.predict(dtest)

        return score, pred, model

    def epoch_log(self, score, i, n_splits):
        print('{}/{}_Fold: val_accuracy: {}'.format(i + 1, n_splits, score))
        logging.info(
            '{}/{}_Fold: val_accuracy: {}'.format(i + 1, n_splits, score))

    def metric(self, va_y, pred):
        score = average_precision_score(va_y, pred)
        return score

    def pr_auc(self, predt, dvalid):
        """カスタム validation metric
        """
        score = average_precision_score(dvalid.get_label(), predt)
        return "pr_auc", score
