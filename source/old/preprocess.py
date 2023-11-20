import sys
import pandas as pd
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from feature_engine import imputation as mdi
from feature_engine import encoding as ce
import warnings
warnings.filterwarnings('ignore')


def get_path():
    cur_path = os.getcwd()
    parent_path = os.path.dirname(cur_path)
    return cur_path, parent_path


def file_path(data_path, file):
    return os.path.abspath(os.path.join(data_path, f'{file}'))


def df_write(data_path, df, file):
    df = df.copy()
    df.to_csv(os.path.abspath(os.path.join(data_path, file)), index=False)


def split_train_test(df, configs):
    df = df.copy()
    X = df.drop(columns=configs['y_col'][0])
    y = df[configs['y_col'][0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=configs['y_col'][0])
    return X_train, X_test, y_train, y_test


def model_selection(option='logic'):
    if option == 'light':
        return lgb.LGBMClassifier(random_state=0)
    else:
        return LogisticRegression(random_state=0)


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

    print('There are {} date_time variables'.format(len(date_time)))
    print('There are {} discrete variables'.format(len(discrete)))
    print('There are {} continuous variables'.format(len(continuous)))
    print('There are {} categorical variables'.format(len(categorical)))

    if y_null:
        data = data[data[Y_col] != data[Y_col].max()].copy()
    else:
        data = data.copy()

    return data, discrete, continuous, categorical


def split_train_test(df, configs):
    df = df.copy()
    X = df.drop(columns=configs['y_col'][0])
    y = df[configs['y_col'][0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=configs['test_size'][0], random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test


def make_imputer_pipe(continuous, discrete, categorical):
    numberImputer = continuous + discrete
    categoricalImputer = categorical

    if (len(numberImputer) > 0) & (len(categoricalImputer) > 0):
        pipe = Pipeline([
            ("median_imputer",
             mdi.MeanMedianImputer(
                 imputation_method="median", variables=numberImputer),),

            ('imputer_cat',
             mdi.CategoricalImputer(variables=categoricalImputer)),

            ('categorical_encoder',
             ce.OrdinalEncoder(encoding_method='ordered',
                               variables=categoricalImputer))
        ])
    else:
        if (len(numberImputer) > 0) & (len(categoricalImputer) == 0):
            pipe = Pipeline([
                ("median_imputer",
                 mdi.MeanMedianImputer(
                     imputation_method="median", variables=numberImputer),)
            ])
        else:
            if (len(numberImputer) == 0) & (len(categoricalImputer) > 0):
                pipe = Pipeline([
                    ('imputer_cat',
                     mdi.CategoricalImputer(variables=categoricalImputer)),

                    ('categorical_encoder',
                     ce.OrdinalEncoder(encoding_method='ordered',
                                       variables=categoricalImputer))
                ])
            else:
                pipe = []
    return pipe


def do_imputation(X_train, X_test, y_train, y_test, pipe):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    if pipe != []:
        pipe.fit(X_train, y_train)
        X_train = pipe.transform(X_train)
        X_test = pipe.transform(X_test)
    else:
        print('no pipe applied')
    return X_train, X_test, y_train, y_test


def do_train(X_train, X_test, y_train, y_test, option):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    model = model_selection(option)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics(y_test, y_pred, option)


def min_max_scale(df):
    df = df.copy()
    scaler = MinMaxScaler()
    scaler.fit(df)
    return scaler.transform(df)


def metrics(y_test, pred, option):
    y_test = y_test.copy()
    pred = pred.copy()
    accuracy = round(accuracy_score(y_test, pred), 2)
    precision = round(precision_score(y_test, pred), 2)
    recall = round(recall_score(y_test, pred), 2)
    f1 = round(f1_score(y_test, pred), 2)
    print(option, "f1 점수:", f1, "정확도:", accuracy, "정밀도:", precision, "재현율:", recall)
    print(confusion_matrix(y_test, pred))


if __name__ == '__main__':
    # arv 예1: credit argumet_credit.xlsx
    # arv 예2: metro argumet_metro.xlsx

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

    # 전처리된 데이터를 지정된 폴더에 저장
    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
    dest_path = os.path.join(parent, os.path.join(f'{dest_path}', f'draft_{ori_file_name}'))
    df_initial.to_csv(dest_path, index=False)

    df, y_null = y_label_enc(df_initial, configs)
    df_organized, discrete, continuous, categorical = organize_data(df, configs, y_null)
    X_train, X_test, y_train, y_test = split_train_test(df_organized, configs)
    pipe = make_imputer_pipe(discrete, continuous, categorical)
    X_train, X_test, y_train, y_test = do_imputation(X_train, X_test, y_train, y_test, pipe)

    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
    dest_path = os.path.join(parent, os.path.join(f'{dest_path}', f'Xtrain_{ori_file_name}'))
    X_train.to_csv(dest_path, index=False)

    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
    dest_path = os.path.join(parent, os.path.join(f'{dest_path}', f'Xtest_{ori_file_name}'))
    X_test.to_csv(dest_path, index=False)

    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
    dest_path = os.path.join(parent, os.path.join(f'{dest_path}', f'ytrain_{ori_file_name}'))
    y_train.to_csv(dest_path, index=False)

    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
    dest_path = os.path.join(parent, os.path.join(f'{dest_path}', f'ytest_{ori_file_name}'))
    y_test.to_csv(dest_path, index=False)

    X_train_scaled = min_max_scale(X_train)
    X_test_scaled = min_max_scale(X_test)
    xtrains = pd.DataFrame(data=X_train_scaled, columns=X_train.columns)
    xtests = pd.DataFrame(data=X_test_scaled, columns=X_test.columns)

    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
    dest_path = os.path.join(parent, os.path.join(f'{dest_path}', f'Scaled_Xtrain_{ori_file_name}'))
    xtrains.to_csv(dest_path, index=False)

    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
    dest_path = os.path.join(parent, os.path.join(f'{dest_path}', f'Scaled_Xtest_{ori_file_name}'))
    xtests.to_csv(dest_path, index=False)

    print('successful Ending')
