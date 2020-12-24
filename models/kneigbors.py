import sys
import os
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split
from torch.utils.data import Dataset
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import trange
from time import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import average_precision_score


class Kneighbor:

    def __init__(self, X, y, X_test, output_path, fold_type, n_slits):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.output_path = output_path
        # self.params
        self.fold_type = fold_type
        self.n_splits = n_slits
        self.search_para = True

    def trainer(self):
        y_pred = np.zeros(len(self.X_test))
        scores = []
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=71)
        if self.fold_type == 'kfold':
            for i, (tr_idx, va_idx) in enumerate(kf.split(self.X)):
                tr_x, va_x = self.X.iloc[tr_idx], self.X.iloc[va_idx]
                tr_y, va_y = self.y.iloc[tr_idx], self.y.iloc[va_idx]

                if self.search_para and i == 0:
                    self.search_parameter(tr_x, tr_y, va_x, va_y)

                score, pred, model = self.train(tr_x, tr_y, va_x, va_y)
                scores.append(score)
                self.epoch_log(score, i, self.n_splits)

                y_pred += pred / self.n_splits

            final_score = sum(scores) / self.n_splits

        elif self.fold_type == 'oof':
            tr_x, va_x, tr_y, va_y = train_test_split(
                self.X, self.y, test_size=0.2)

            if self.search_para:
                self.search_parameter(tr_x, tr_y, va_x, va_y)

            score, pred, model = self.train(tr_x, tr_y, va_x, va_y)
            scores.append(score)
            self.epoch_log(score, 0, 1)
            y_pred += pred
            final_score = score

        print('avarage_accuracy: {}'.format(final_score))
        logging.info('avarage_accuracy: {}'.format(final_score))
        return y_pred

    def train(self, tr_x, tr_y, va_x, va_y):

        model = KNeighborsClassifier(n_neighbors=6)
        model.fit(tr_x, tr_y)
        y_val_pred = model.predict(va_x)
        score = self.metric(va_y, y_val_pred)

        pred = model.predict(self.X_test)

        return score, pred, model

    def epoch_log(self, score, i, n_splits):
        print('{}/{}_Fold: val_accuracy: {}'.format(i + 1, n_splits, score))
        logging.info(
            '{}/{}_Fold: val_accuracy: {}'.format(i + 1, n_splits, score))

    def metric(self, va_y, pred):
        score = average_precision_score(va_y, pred)
        return score

    def search_parameter(self, tr_x, tr_y, va_x, va_y):
        accuracy_list = []
        sns.set()
        k_range = range(1, 5)
        for k in k_range:
            print("start: {}".format(k))
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(tr_x, tr_y)
            va_pred = knn.predict(va_x)
            accuracy_list.append(accuracy_score(va_y, va_pred))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(k_range, accuracy_list)
        ax.set_xlabel('k-nn')
        ax.set_ylabel('accuracy')
        fig.savefig(self.output_path + '/knn_param.png')
