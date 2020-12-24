import sys
import re
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

from FE.categorical import *
from FE.numerical import *
from FE.missing import *
from FE.datetime import *
from FE.nlp.embeddings import *
from FE.nlp.nlp import *
from FE.dem_del import *


def split_0(x):
    try:
        a = x.split(':')
        return a[0]
    except:
        return None


def split_1word(x):
    try:
        a = x.split(' ')
        return a[0]
    except:
        return None
 

def preprocess(train, test, target, model_name):
    print('Start preprocess')
    """特徴量生成
    """
    df = pd.concat([train, test], axis=0)
    df['Name'] = clearn_by_hero(df['Name'])
    # df = get_n_gram_feature(df, 2, 'Name')
    df = get_tfidf_pca(df, 'Name', 5)
    # df = create_boolean_feature(df)
    df = get_sales_portfolio(df, 'Genre')
    df = get_sales_portfolio(df, 'Rating')
    # df['kmeans_Name'] = df_name['kmeans']
    df['Name_0'] = df['Name'].apply(split_0)
    df['Name_2'] = df['Name'].apply(split_1word)
    df['Name_len'] = df['Name'].apply(lambda x: len(str(x)))

    df['P_G'] = df['Platform'].fillna('Nothing') + df['Genre'].fillna('Nothing')
    
    """category変数のエンコーディング
    """
    label_cat_cols = ['Platform', 'Rating']
    freq_cat_cols = ['Name_0', 'Name_2', 'Genre', 'Rating', 'Developer', 'P_G']
    cat_cols = ['Name_0', 'Name_2', 'Platform', 'Genre', 'Rating', 'Developer', 'P_G']
    sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    difficult_cols = ['Publisher']

    df = frequency_encoding(df, freq_cat_cols)
    df = label_encoding(df, freq_cat_cols)

    df = missing_mean_2(df, ['User_Score'])

    """Publisherの活用
    """
    # df = pivot_pca(df, 5, 'Publisher', ["Platform"])
    # df = pivot_pca(df, 5, 'Publisher', ["Genre"])
    df = pivot_pca_some(df, 7, 'Publisher', ["Year_of_Release", "Platform", "Genre"])
    df = pivot_pca_some(df, 7, 'Developer', ["Year_of_Release", "Platform", "Genre"])
    df = pivot_pca_some(df, 7, 'Year_of_Release', ["Name_0", "P_G"])
    # df = pivot_pca(df, 5, 'Developer', ["Genre"])
    # df = pivot_pca(df, 5, 'Developer', ["Year_of_Release"])

    df = n_gram_PCA(df, 'Name', 5)

    """simple EDA
    """
    df['User_score_count'] = df['User_Score'] * df['User_Count']
    df['Critic_Score_count'] = df['Critic_Score'] * df['Critic_Count']

    """trainの高い値を削除
    """
    df = trans_log1p(df, 'Global_Sales')

    # print(df.dtypes)
    statistics_cols = ['User_Score', 'User_Count', 'Critic_Score']
    df = group_statitics(df, ['Platform'], statistics_cols)
    df = group_statitics(df, ['Genre'], statistics_cols)
    df = group_statitics(df, ['Year_of_Release'], statistics_cols)

    id_list = [['Genre', 'Platform'], ['Name_0_freq', 'Critic_Count']]
    df = row_count(df, id_list, ['Name'])
    df = count_tier(df, 'Platform')
    df = count_tier(df, 'Year_of_Release')
    df = count_tier(df, 'Developer')
    df = count_tier(df, 'Publisher')


    """不要カラム削除
    """
    df = df.drop('Name', axis=1)
    df = df.drop(cat_cols, axis=1)
    df = df.drop(difficult_cols, axis=1)
    df = df.drop(sales_columns, axis=1)

    if model_name == 'nn':
        df = missing_median(df, df.columns)

    for i, j in zip(df.isnull().sum(), df.isnull().sum().index):
        print(i, j)

    del_col = [
        # 'User_Score_Year_of_Release_min',
        # 'Critic_Score_Platform_sum',
        # 'User_Count_Year_of_Release_sum',
        # 'SP@Rating_NA_Sales',
        # 'SP@Rating_EU_Sales',
        # 'SP@Rating_Other_Sales',
        # 'SP@Rating_JP_Sales',        
        # 'SP@Genre_Other_Sales'
        # 'User_Score_Genre_mean'
        # 'SP@Genre_JP_Sales'
        # 'PCA_Name_n_gram_PCA_4'
        # 'Critic_Score_Platform_max',
        # 'PCA_Name_n_gram_PCA_1',
        # 'PCA_Name_n_gram_PCA_3',
        # 'Critic_Score_Genre_sum',
        # 'PCA_Name_n_gram_PCA_0',
        # 'User_Count_Genre_sum',
        # 'PCA_Name_n_gram_PCA_2',
        # 'User_Count_Genre_mean',
        # 'User_Score_Platform_max',
        # 'Critic_Score_Genre_max',
        # 'User_Score_Genre_max',
        # 'Rating_label',
        'Rating_freq',
        'User_Score_Genre_sum',
        'count_tier_Publisher',
        'Critic_Score_Genre_min',
        'count_tier_Developer',
        'count_tier_Platform',
        'User_Count_Year_of_Release_min',
        'count_tier_Year_of_Release',
        'User_Count_Platform_min',
        'User_Count_Genre_min',
        'PCA_Publisher_mix_1',
        'PCA_Publisher_mix_0']
    
    df = df.drop(del_col, axis=1)

    print('Finish preprocess')

    return df
