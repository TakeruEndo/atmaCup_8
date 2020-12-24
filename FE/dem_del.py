import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.decomposition import PCA, TruncatedSVD


def apply_pca(df, targets):
    """主成分分析
    標準化を事前に行っておく必要がある
    """
    pca = PCA(n_components=10)
    pca.fit(df[targets])
    # 変換の適用
    pca_df = pca.transform(df[targets])
    # 主成分得点
    pd.DataFrame(pca_df, columns=["PCA_{}".format(x + 1)
                                  for x in range(len(df[targets].columns))])
    # 第一主成分と第二主成分でプロットする
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(pca_df[:, 0], pca_df[:, 1], alpha=0.8, c=list(df.iloc[:, 0]))
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    fig.savefig("outputs/pca.png")
    return np.concat([df, pca_df], axis=1)


def apply_pca_2(df, targets):
    """主成分分析
    標準化を事前に行っておく必要がある
    """
    pca = PCA(n_components=5)
    pca.fit(df[targets])
    # 変換の適用
    pca_df = pca.transform(df[targets])
    # 主成分得点
    pca_df = pd.DataFrame(pca_df, columns=["PC{}".format(x + 1)
                                  for x in range(len(df[targets].columns))])
    return pca_df 


def apply_tsvd(df, targets):
    """TrunatedSVD
    疎行列を扱える
    """
    svd = TruncatedSVD(n_components=5, random_seed=71)
    svd.fit(df[targets])
    svd_df = svd.transform(df[targets])
    pd.DataFrame(svd_df, columns=["PC{}".format(x + 1)
                                  for x in range(len(df[targets].columns))])
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(svd_df[:, 0], svd_df[:, 1], alpha=0.8, c=list(df.iloc[:, 0]))
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    fig.savefig("outputs/tsvd.png")
    return np.concat([df, svd_df], axis=1)
