import os
import pandas as pd
import numpy as np
import datetime


dt_now = datetime.datetime.now()
model_name = 'ensemble1'
# ログレベルを DEBUG に変更
output_path = 'outputs/{}_{}-{}-{}-{}'.format(model_name, str(
    dt_now.year), str(dt_now.day), str(dt_now.hour), str(dt_now.minute))
os.mkdir(output_path)


if __name__ == '__main__':
    dir_paths = ['lgbm_2020-10-21-51', 'cat_2020-13-15-15']
    for i, v in enumerate(dir_paths):
        sub = pd.read_csv('outputs/{}/submission_{}.csv'.format(v, v))
        if i == 0:
            sub_mix = sub.Global_Sales.values * 0.65
        elif i == 1:
            sub_mix += sub.Global_Sales.values * 0.35
        # elif i == 2:
        #     sub_mix += sub.OV_J3_sum.values * 0.15

    submit_df = pd.DataFrame({'Global_Sales': sub_mix})
    submit_df.to_csv(output_path + '/submission_{}_{}-{}-{}-{}.csv'.format(model_name, str(
        dt_now.year), str(dt_now.day), str(dt_now.hour), str(dt_now.minute)), index=False)
