from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(X):
    tr_x, va_x = X.iloc[tr_idx], X.iloc[va_idx]
    tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]
