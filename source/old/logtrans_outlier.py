import sys
import pandas as pd
import numpy as np
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


def metrics(y_test, pred, option, display_confusion=False):
    y_test = y_test.copy()
    pred = pred.copy()
    accuracy = round(accuracy_score(y_test, pred), 2)
    precision = round(precision_score(y_test, pred), 2)
    recall = round(recall_score(y_test, pred), 2)
    f1 = round(f1_score(y_test, pred), 2)
    print(option, "f1 점수:", f1, "정확도:", accuracy, "정밀도:", precision, "재현율:", recall)
    if display_confusion:
        print(confusion_matrix(y_test, pred))


def drop_outlier(df=None, corr_highest=None, y_col=None, yes_value=None, weight=1.5):
    df = df.copy()
    targeted = df[df[y_col]==yes_value][corr_highest]
    quantile_25 = np.percentile(targeted.values, 25)
    quantile_75 = np.percentile(targeted.values, 75)

    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight

    outlier_index = targeted [(targeted  < lowest_val) | (targeted > highest_val)].index
    df = df.drop(outlier_index, axis=0)
    return df


def log_trans(df, trans_col):
    df = df.copy()
    trans_values = np.log1p(df[trans_col])
    df.drop([trans_col], axis=1, inplace=True)
    df.insert(0, trans_col, trans_values)
    return df

#     if outlier_process:
# #         outlier_index = get_outlier(df_copy, corr_higher, y_col, weight=1.5)
# #         df_copy.drop(outlier_index, axis=0, inplace=True)
#         df = get_outlier(df, corr_higher, y_col, yes_value, weight=1.5)
#     return df


def get_corr_top5(df, y_col, yes_value):
    df = df.copy()
    corr_mat = df.corr(method='pearson')
    upper_corr_mat = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(np.bool))

    # Convert to 1-D series and drop Null values
    unique_corr_pairs = upper_corr_mat.unstack().dropna()

    # Sort correlation pairs
    sorted_mat = unique_corr_pairs.sort_values()
    df_corr = pd.DataFrame(data=sorted_mat).reset_index()
    df_corr = df_corr[df_corr['level_0']==y_col]
    df_corr = df_corr.rename(columns={'level_0':'Y_col', 'level_1':'col', 0:'corr_value'})
    df_corr['abs_corr_value'] = abs(df_corr['corr_value'])
    df_corr = df_corr.sort_values(by=['abs_corr_value'], ascending=False).reset_index(drop=True)
    corr_higher = df_corr.iloc[0:5]
    return corr_higher


def split_impute_train(df, configs, discrete, continuous, categorical, option='logic'):
    X_train, X_test, y_train, y_test = split_train_test(df, configs)
    pipe = make_imputer_pipe(discrete, continuous, categorical)
    X_train, X_test, y_train, y_test = do_imputation(X_train, X_test, y_train, y_test, pipe)
    do_train(X_train, X_test, y_train, y_test, option)

dataset = 1

if dataset == 0:
    folder_name = 'credit'
    config_file_name = 'argumet_credit.xlsx'
else:
    folder_name = 'metro'
    config_file_name = 'argumet_metro.xlsx'

cur_path = os.getcwd()
parent = os.path.abspath(os.path.join(cur_path, os.pardir))
config_file = os.path.join(parent, os.path.join('config', f'{config_file_name}'))
configs = pd.read_excel(config_file, header=None).set_index(0).T
configs = configs.to_dict('list')
ori_file_name = configs['file_name'][0]
configs['file_name'][0] = os.path.join(parent, os.path.join('data', configs['file_name'][0]))
Y_col = configs['y_col'][0]
df_initial = read_data(configs)

df, y_null = y_label_enc(df_initial, configs)
df_organized, discrete, continuous, categorical = organize_data(df, configs, y_null)

split_impute_train(df_organized, configs, discrete, continuous, categorical, option='logic')

if folder_name == 'credit':
    trans_col = 'Amount'
else:
    trans_col = 'DV_pressure'

df_log_processed = log_trans(df_organized, trans_col)
split_impute_train(df_log_processed, configs, discrete, continuous, categorical, option='logic')

yes_value = 1

corr_higher = get_corr_top5(df_organized, Y_col, yes_value)

for idx, row in corr_higher.iterrows():
    print(idx, row['col'], row['corr_value'])
    corr_highest = row[1]
    df_processed = log_trans(df_organized, trans_col)
    df_processed = drop_outlier(df_processed, corr_highest, Y_col, yes_value, weight=1.5)
    split_impute_train(df_processed, configs, discrete, continuous, categorical, option='logic')
    print()