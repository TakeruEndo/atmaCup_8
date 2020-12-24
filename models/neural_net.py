import sys
import os
import copy
import logging
from glob import glob
from time import time
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import trange
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from preprocess import preprocess
from models.models import CustomLinear, MLP_Model

from models.utils import save_feature_impotances, save_params


def seed_torch(seed=1029):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NeuralNet:
    def __init__(self, X, y, X_test, output_path, fold_type, n_splits):
        self.X = X
        self.y = y
        self.X_test = X_test
        self.output_path = output_path
        self.fold_type = fold_type
        self.n_splits = n_splits
        self.batch_size = 32
        self.train_epochs = 15
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else 'cpu')

    def trainer(self):
        seed_torch(1030)
        print(self.X.shape, self.X_test.shape)
        whole_data = pd.concat([self.X, self.X_test], axis=0)
        scaler = StandardScaler()
        whole_data = scaler.fit_transform(whole_data)

        X = whole_data[:len(self.X)]
        test = whole_data[len(self.X):]
        y = self.y.values

        # testのdataloaderを定義
        test = torch.from_numpy(test.astype(np.float32))
        test_dataset = torch.utils.data.TensorDataset(test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False)

        # modelを定義
        net = MLP_Model(len(self.X.columns), 1, 80)

        test_pred_all = np.zeros(len(test))

        # kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=71)
        # for i, (train_idx, valid_idx) in enumerate(kf.split(X)):
        num_bins = np.int(1 + np.log2(len(self.X)))
        bins = pd.cut(
            self.y,
            bins=num_bins,
            labels=False
        )
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=71)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X=self.X, y=bins.values)):

            x_train_fold = torch.from_numpy(X[train_idx].astype(np.float32))
            y_train_fold = torch.from_numpy(
                y[train_idx, np.newaxis].astype(np.float32))
            x_val_fold = torch.from_numpy(X[valid_idx].astype(np.float32))
            y_val_fold = torch.from_numpy(
                y[valid_idx, np.newaxis].astype(np.float32))

            model = net.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5)
            criterion = nn.MSELoss()

            train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
            valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
            train_loader = torch.utils.data.DataLoader(
                train, batch_size=self.batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(
                valid, batch_size=self.batch_size, shuffle=True)

            print('----------')
            print(f'Start fold {fold}/{self.n_splits}')
            print('----------')

            best_val_loss = 10 * 5
            # epoch分のループを回す
            for epoch in range(self.train_epochs):
                model.train()
                avg_loss = 0
                y_preds_list = np.array([])
                y_batch_list = np.array([])

                for x_batch, y_batch in train_loader:
                    preds = model(x_batch)
                    loss = criterion(preds, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # accuracyの計算
                    y_preds = preds.detach().numpy().copy()
                    y_batch = y_batch.detach().numpy().copy()
                    # 足す
                    y_preds_list = np.append(y_preds_list, y_preds)
                    y_batch_list = np.append(y_batch_list, y_batch)
                    avg_loss += loss.item() / len(train_loader)

                y_preds_list = y_preds_list.clip(0)
                y_preds_list = np.expm1(y_preds_list)
                y_batch_list = np.expm1(y_batch_list)
                accuracy = self.metric(y_batch_list, y_preds_list)
                scheduler.step()
                model.eval()
                avg_val_loss = 0
                y_val_preds_list = np.array([])
                y_val_batch_list = np.array([])

                for i, (x_batch, y_batch) in enumerate(valid_loader):
                    preds = model(x_batch)
                    loss = criterion(preds, y_batch)
                    y_preds = preds.detach().numpy().copy()
                    y_batch = y_batch.detach().numpy().copy()
                    # 足す
                    y_val_preds_list = np.append(y_val_preds_list, y_preds)
                    y_val_batch_list = np.append(y_val_batch_list, y_batch)
                    avg_val_loss += loss.item() / len(valid_loader)

                if best_val_loss > avg_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), os.path.join(
                        self.output_path, f"fold_{fold}_{epoch}.pth"))
                y_val_preds_list = y_val_preds_list.clip(0)
                y_val_preds_list = np.expm1(y_val_preds_list)
                y_val_batch_list = np.expm1(y_val_batch_list)
                val_accuracy = self.metric(y_val_batch_list, y_val_preds_list)

                print('Epoch {}/{} \t loss={:.4f} \t score={:.4f} \t val_loss={:.4f} \t val_score={:.4f} '.format(
                    epoch + 1, self.train_epochs, avg_loss, accuracy, avg_val_loss, val_accuracy))
                logging.info('Epoch {}/{} \t loss={:.4f} \t score={:.4f} \t val_loss={:.4f} \t val_score={:.4f} '.format(
                    epoch + 1, self.train_epochs, avg_loss, accuracy, avg_val_loss, val_accuracy))

            test_pred = np.array([])
            test_dataset = torch.utils.data.TensorDataset(test)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False)

            test_pred = self.test(test_loader, fold)
            test_pred_all += test_pred

        test_pred_all = test_pred_all / self.n_splits
        test_pred_all = np.expm1(test_pred_all)
        return test_pred_all

    def test(self, test_loader, fold):
        weight_path = self.get_weight(fold)
        net = MLP_Model(len(self.X.columns), 1, 80)
        net.load_state_dict(torch.load(weight_path))
        net = net.to(self.device)
        # X_test_fold をbatch_sizeずつ渡すループ
        test_pred = np.array([])
        for i, (x_batch, ) in enumerate(test_loader):
            y_pred = net(x_batch)
            y_pred = y_pred.detach().numpy().copy()
            test_pred = np.append(test_pred, y_pred)
        return test_pred

    def get_weight(self, fold):
        weight_list = list()
        weight_list += glob(os.path.join(self.output_path, f"fold_{fold}_?.pth"))
        weight_list += glob(os.path.join(self.output_path, f"fold_{fold}_??.pth"))
        return weight_list[-1]

    def metric(self, va_y, pred):
        try:
            score = np.sqrt(mean_squared_log_error(va_y, pred))
        except:
            score = 0
        return score
