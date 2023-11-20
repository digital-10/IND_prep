import sys
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def get_path():
    cur_path = os.getcwd()
    parent_path = os.path.dirname(cur_path)
    return cur_path, parent_path


def file_path(data_path, file):
    return os.path.abspath(os.path.join(data_path, f'{file}'))


def read_data(configs):
    if configs['date_col'][0] == ' ':
        df = pd.read_csv(configs['file_name'][0])
    else:
        df = pd.read_csv(configs['file_name'][0], parse_dates=configs['date_col'])

    if configs['remove_col'][0] == ' ':
        pass
    else:
        if configs['remove_col'][0] in df.columns.to_list():
            df = df.drop(configs['remove_col'][0], axis=1)

    return df


def y_label_enc(df, configs):
    df = df.copy()
    Y_col = configs['y_col'][0]
    if df[Y_col].isnull().any():
        Y_null = True
    else:
        Y_null = False
    labeler = LabelEncoder()
    df[Y_col] = labeler.fit_transform(df[Y_col])
    return df, Y_null


def organize_data(df, configs, y_null):
    df = df.copy()
    cols = df.columns.to_list()
    null_threshhold_cols = []
    no_null_cols = []
    date_time = configs['date_col']
    Y_col = configs['y_col'][0]

    for col in cols:
        null_mean = df[col].isnull().mean()
        if null_mean >= configs['null_threshhold'][0]:
            null_threshhold_cols.append(col)
        if null_mean == 0:
            no_null_cols.append(col)

    cols_stayed = [item for item in cols if item not in null_threshhold_cols]
    data = df[cols_stayed].copy()

    # numerical: discrete vs continuous
    discrete = [var for var in cols_stayed if
                data[var].dtype != 'O' and var != Y_col and var not in date_time and data[var].nunique() < 10]
    continuous = [var for var in cols_stayed if
                  data[var].dtype != 'O' and var != Y_col and var not in date_time and var not in discrete]

    # categorical
    categorical = [var for var in cols_stayed if data[var].dtype == 'O' and var != Y_col]

    print('There are {} date time variables'.format(len(date_time)))
    print('There are {} discrete variables'.format(len(discrete)))
    print('There are {} continuous variables'.format(len(continuous)))
    print('There are {} categorical variables'.format(len(categorical)))

    if y_null:
        data = data[data[Y_col] != data[Y_col].max()].copy()
    else:
        data = data.copy()

    return data, discrete, continuous, categorical


if __name__ == '__main__':
    # arv 예1: credit argumet_credit.xlsx
    # arv 예2: metro argumet_metro.xlsx
        
    try:
        folder_name = sys.argv[1]
        config_file_name = sys.argv[2]
        cur_path = os.getcwd()
        parent = os.path.abspath(os.path.join(cur_path, os.pardir))
        config_file = os.path.join(parent, os.path.join('config', f'{config_file_name}'))
        configs = pd.read_excel(config_file, header=None).set_index(0).T
        configs = configs.to_dict('list')
        ori_file_name = configs['file_name'][0]
        configs['file_name'][0] = os.path.join(parent, os.path.join('data', configs['file_name'][0]))        
        df_initial = read_data(configs)

        # 전처리 저장 경로 정의
        dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
        dest_path = os.path.join(parent, os.path.join(f'{dest_path}/y_enc', f'{ori_file_name}_y_enc.csv'))

        # Label 칼럼 인코딩
        df, y_null = y_label_enc(df_initial, configs)
        df_organized, discrete, continuous, categorical = organize_data(df, configs, y_null)
        
        # 인코딩 데이터셋 저장
        df_organized.to_csv(dest_path, index=False)        
        print('Completed.')

    except Exception as e:
        print('에러발생', e)

    