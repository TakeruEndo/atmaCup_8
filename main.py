import sys
import os
import copy
import logging
import shutil
from time import time
import torch
import random
from datetime import datetime

import pandas as pd
import numpy as np

from tqdm.notebook import trange


from preprocess import preprocess
from utils import select_model
from models import lightGBM


def seed_torch(seed=1029):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def submit_csv(pred):
    # 提出用に整形
    submit_df = pd.DataFrame({'Global_Sales': pred})
    submit_df.to_csv(output_path + '/submission_{}_{}-{}-{}-{}.csv'.format(model_name, str(
        dt_now.year), str(dt_now.day), str(dt_now.hour), str(dt_now.minute)), index=False)


def submit_csv_dict(pred_dict):
    # 提出用に整形
    for k, v in pred_dict.items():
        submit_df = pd.DataFrame({'Global_Sales': v})
        submit_df.to_csv(output_path + '/{}_submission_{}_{}-{}-{}-{}.csv'.format(k, model_name, str(
            dt_now.year), str(dt_now.day), str(dt_now.hour), str(dt_now.minute)), index=False)


if __name__ == '__main__':

    seed_torch(42)
    dt_now = datetime.now()
    model_name = 'lgbm'
    fold_type = 'skfold'
    n_splits = 5
    target = 'Global_Sales'
    output_path = 'outputs/{}_{}-{}-{}-{}'.format(model_name, str(
        dt_now.year), str(dt_now.day), str(dt_now.hour), str(dt_now.minute))
    os.mkdir(output_path)

    logging.basicConfig(filename=os.path.join(
        output_path, 'logger.log'), level=logging.INFO)

    # preprocessとmainを複製
    shutil.copyfile("./preprocess.py",
                    os.path.join(output_path, 'preprocess.py'))
    shutil.copyfile("./main.py", os.path.join(output_path, 'main.py'))

    ROOT = "inputs"

    train = pd.read_csv(f"{ROOT}/train.csv")
    test = pd.read_csv(f"{ROOT}/test.csv")

    """重複行を削除
    keep: False <= 重複した行を全て削除 (defaultはTrue)
    inplace: True <= 元のデータセットから削除
    subset: 行を選択 (subset=['Patient', 'Weeks'])
    """
    train.drop_duplicates(inplace=True)

    """trainとtestの共通データを取り出す
    """
    # train = train[train['目的のカラム'].isin(test.{目的のカラム}.unique())]

    data = preprocess(train, test, target, model_name)

    print('columns_length: ', len(data.columns))
    train = data[: len(train)]
    test = data[len(train):]

    """モデルとパラメータの設定
    """
    # TODO: txtファイルで重み保存フォルダに一緒に保存

    """学習と推論
    """
    train_x = train.drop(target, axis=1)
    y = train[target]
    test = test.drop(target, axis=1)
    model = select_model(model_name, train_x, y, test, output_path, fold_type, n_splits)
    pred = model.trainer()

    # いろいろ推論
    # model_name = 'cat'
    # fold_type = 'skfold'
    # n_splits = 4
    # target = 'Global_Sales'

    # test_sales = pd.DataFrame()
    # special_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    # for target in special_cols:
    #     if target == 'Global_Sales':
    #         continue
    #     train_x = train.drop(special_cols, axis=1)
    #     y = train[target]
    #     y = np.log1p(y)
    #     test_ = test.drop(special_cols, axis=1)
    #     model = select_model(model_name, train_x, y, test_, output_path, fold_type, n_splits)
    #     pred = model.trainer()
    #     test_sales[target] = pred
    #     print(f'---finish_{target}_pred----')

    # """最後のpredict
    # """
    # model_name = 'lgbm'
    # fold_type = 'skfold'
    # n_splits = 4
    # target = 'Global_Sales'
    # train_x = train.drop(target, axis=1)
    # y = train[target]
    # test = test.drop(special_cols, axis=1)
    # test = pd.concat([test, test_sales], axis=1)
    # model = select_model(model_name, train_x, y, test, output_path, fold_type, n_splits)
    # pred = model.trainer()

    """提出
    """
    if model_name == 'linear':
        submit_csv_dict(pred)
    else:
        # 出力フォルダの作成
        submit_csv(pred)


