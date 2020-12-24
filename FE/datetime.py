import pandas as pd
import datetime


def get_date_time(df, target):
    # datetime型に変換
    to_datetime = pd.to_datetime(df[target], format='%Y-%m-%d %H:%M:%S')
    base_time = datetime.datetime(2020, 4, 28)
    df['total_seconds'] = to_datetime.map(
        lambda x: (x - base_time).total_seconds())
    df['year'] = to_datetime.map(lambda x: x.year)
    df['month'] = to_datetime.map(lambda x: x.month)
    df['day'] = to_datetime.map(lambda x: x.day)
    df['hour'] = to_datetime.map(lambda x: x.hour)
    df['minute'] = to_datetime.map(lambda x: x.minute)
    df['second'] = to_datetime.map(lambda x: x.second)
    df['dayofweek'] = to_datetime.map(lambda x: x.dayofweek)

    df = df.drop([target], axis=1)
    return df
