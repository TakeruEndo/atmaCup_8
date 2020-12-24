import re
import numpy as np


def missing_mean(df, columns):
    for c in columns:
        df[c] = df[c].fillna(np.nanmean(df[c].values))
    return df


def missing_median(df, columns):
    for c in columns:
        df[c] = df[c].fillna(np.nanmedian(df[c].values))
    return df


def missing_mean_2(df, columns):
    """
    型変換と欠損値の両方を処理したい場合
    """
    for c in columns:
        # 仮
        copy = df[c].copy()
        mean = np.nanmean(copy.replace(
            'tbd', np.nan).dropna().astype(np.float64))
        # 適用
        df[c] = df[c].replace('tbd', np.nan)
        df[c] = df[c].fillna(mean)
        df[c] = df[c].astype(np.float64)
    return df


def rename_columns(df, dict_name):
    """columns名前変換
    new_name = {元の名前: 変換後, 元の名前: 変換後}
    """
    new_name = dict_name
    df.rename(new_name, axis=1, inplace=True)
    return df


def sub_number(x):
    """数字以外を除去
    """
    return re.sub('[^0-9.]', '', x)
