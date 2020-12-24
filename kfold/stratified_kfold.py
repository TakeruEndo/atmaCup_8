from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd


"""連続値に対する層化抽出
"""
data = dataset_df.reset_index(drop=True)

data["kfold"] = -1

# ビンの計算
num_bins = np.floor(1 + np.log2(len(data)))

# bin targets
data.loc[:, "bins"] = pd.cut(
    data['Water Solubility'].values, int(num_bins), labels=False
)

# initialize
kf = StratifiedKFold(n_splits=5, shuffle=True)

# fill the new kfold columns
# note that, instead of targets, we use bins!
for f, (T_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
    data.loc[v_, 'kfold'] = f

data = data.drop('bins', axis=1)

"""離散値に関する層化抽出
"""
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x, train_y):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
