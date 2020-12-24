from sklearn.model_selection import KFold

patient_id = train_df['Patient']
unique_id = patient_id.unique()
fold = 0

# Fflodで、ID単位で分割する
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_group_idx, va_group_idx in kf.split(unique_id):
    print("--------------------------")
    print('{}_Folds_Start'.format(fold))

    tr_groups, va_groups = unique_id[tr_group_idx], unique_id[va_group_idx]

    is_tr = patient_id.isin(tr_groups)
    is_va = patient_id.isin(va_groups)
    tr_x = train_x[is_tr]
    va_x = train_x[is_va]
    tr_y = train_y[is_tr]
    va_y = train_y[is_va]