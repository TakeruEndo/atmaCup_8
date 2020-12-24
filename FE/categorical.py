import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA, TruncatedSVD


def one_hot_encoding(df, cat_cols):
    """ワンホット表現
    """
    ohe = OneHotEncoder(sparse=False, categories='auto')
    for c in cat_cols:
        df[c] = df[c].fillna('missing')
    ohe.fit(df[cat_cols])

    # ダミー変数の列名の作成
    columns = []
    for i, c in enumerate(cat_cols):
        columns += [f'{c}_{v}' for v in ohe.categories_[i]]

    # ダミー変数をデータフレームに変換
    dummy_vals = pd.DataFrame(ohe.transform(df[cat_cols]), columns=columns)
    dummy_vals = dummy_vals.reset_index(drop=True)
    df = df.reset_index(drop=True)
    df = pd.concat([df, dummy_vals], axis=1)
    return df


def label_encoding(df, cat_cols):
    """
    しっかり最後にカラムを消すこと
    """
    for c in cat_cols:
        values = df[c].copy().fillna('missing')
        le = LabelEncoder()
        le.fit(values)
        df[c + '_label'] = le.transform(values)
        # df[c] = le.transform(values)
    return df


def frequency_encoding(df, cat_cols):
    for c in cat_cols:
        values = df[c].copy().fillna('missing')
        freq = values.value_counts()
        # カテゴリの出現回数で置換
        df[c + '_freq'] = values.map(freq)
    return df


def target_encoding(train, test, cat_cols, target):
    """
    CVを合わせないとリークして大変になる....
    """
    # 変数をループしてtarget encoding
    for c in cat_cols:
        train[c] = train[c].fillna('missing')
        test[c] = test[c].fillna('missing')
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp = pd.DataFrame({c: train[c], 'target': train[target]})
        target_mean = data_tmp.groupby(c)['target'].mean()
        # テストデータのカテゴリを置換
        test[c + '_target'] = test[c].map(target_mean)

        # 学習データの変換後の値を格納する配列を準備
        tmp = np.repeat(np.nan, train.shape[0])
        y = train['Global_Sales']
        # 学習データを分割
        num_bins = np.int(1 + np.log2(len(train)))
        bins = pd.cut(
            y,
            bins=num_bins,
            labels=False
        )
        kf = StratifiedKFold(n_splits=5,
                             shuffle=True, random_state=71)
        for i, (tr_idx, va_idx) in enumerate(kf.split(X=train, y=bins.values)):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp.iloc[tr_idx].groupby(c)['target'].mean()
            # 変換後の値を一時に配列を格納
            tmp[va_idx] = train[c].iloc[va_idx].map(target_mean)

        # 変換後のデータで元の変数を置換
        train[c + '_target'] = tmp

    return train, test


def pivot_pca(df, n_components, target, columns):
    """pivot tableを使ってPCAする
    Args:
        n_components ([type]): PCA
        target ([type]): エンコーディングしたいカラム
        columns ([type]): 数えられるカテゴリ
    """
    plat_pivot = df.pivot_table(index=target, columns=columns, values='Name', aggfunc='count').reset_index()
    plat_pivot_df = plat_pivot[[target]]
    pca = PCA(n_components=n_components)
    features = plat_pivot[[i for i in plat_pivot.columns if i != target]]
    features = features.fillna(-1)
    pca.fit(features)
    # 変換の適用
    pca_df = pca.transform(features)
    # 主成分得点
    for i in range(n_components):
        plat_pivot_df[f"PCA_{target}_{columns[0]}_{i}"] = pca_df[:, i]
    df = pd.merge(df, plat_pivot_df, on=target, how='left')
    return df


def pivot_pca_some(df, n_components, target, column_list):
    """pivot tableを使ってPCAする
    Args:
        n_components ([type]): PCA
        target ([type]): エンコーディングしたいカラム
        columns ([type]): 数えられるカテゴリ
    """
    new_df = pd.DataFrame()
    for c in column_list:
        plat_pivot = df.pivot_table(index=target, columns=c, values='Name', aggfunc='count').reset_index()
        plat_pivot_df = plat_pivot[[target]]
        plat_pivot = plat_pivot.drop(target, axis=1)
        new_df = pd.concat([new_df, plat_pivot], axis=1)
    pca = PCA(n_components=n_components)
    new_df = new_df.fillna(-1)
    pca.fit(new_df)
    # 変換の適用
    pca_df = pca.transform(new_df)
    # 主成分得点
    for i in range(n_components):
        plat_pivot_df[f"PCA_{target}_mix_{i}"] = pca_df[:, i]
    df = pd.merge(df, plat_pivot_df, on=target, how='left')
    return df
