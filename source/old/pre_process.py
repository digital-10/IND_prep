import sys
import pandas as pd
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb

from feature_engine import imputation as mdi
from feature_engine import encoding as ce
from feature_engine import outliers as ol


def get_path():
    cur_path = os.getcwd()
    parent_path = os.path.dirname(cur_path)
    return cur_path, parent_path


def write_processed(train, test, naming, drop):
    _, parent_path = get_path()
    df = train.copy()
    del train
    df[Y_col] = test.to_list()
    del test
    df.to_csv(f'{parent_path}/result_data/{naming}.csv', index=drop)


def model_selection(option):
    if option == 'light':
        return lgb.LGBMClassifier(random_state=0)
    elif option == 'boost':
        return GradientBoostingClassifier(random_state=0)
    else:
        return LogisticRegression(random_state=0)

def metrics(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
#     roc_score = roc_auc_score(y_test,pred,average='macro')
    print('정확도 : {0:.2f}, 정밀도 : {1:.2f}, 재현율 : {2:.2f}'.format(accuracy, precision, recall))
    print('f1-score : {0:.2f}'.format(f1))
#     print('f1-score : {0:.2f}, auc : {1:.2f}'.format(f1,roc_score,recall))

def do_imputation(X_train, X_test, y_train, y_test, pipe):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    if pipe != []:
        pipe.fit(X_train, y_train)
        X_train_t = pipe.transform(X_train)
        X_test_t = pipe.transform(X_test)
    else:
        print ('no pipe')
    return X_train, X_test, y_train, y_test


def do_train(X_train, X_test, y_train, y_test, option):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    model = model_selection(option)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(option)
    print('f1 score macro', f1_score(y_test, y_pred, average='macro'))
    metrics(y_test, y_pred)
    # write_processed(X_test_t, y_test, 'X_test_transform', drop)
    # write_processed(X_train_t, y_train, 'X_train_transform', drop)

def read_config(config_name):
    import yaml
    try:
        _, parent_path = get_path()
        with open(f'{parent_path}/yml/{config_name}', "r", encoding="utf-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            filename = cfg['file']
            null_threshhold = cfg['null_threshhold']
            FOLD = cfg['FOLD']
            TEST_SIZE = cfg['TEST_SIZE']
            Y_col = cfg['Y_col']
            mixed = cfg['mixed']
            date_type = cfg['date_type']
            # date_remove = cfg['date_remove']
            col_remove = cfg['col_remove']
            y_col_is_null = cfg['y_col_is_null']
            uids = cfg['uids']
            drop = cfg['drop']
            option = cfg['option']
            return filename, null_threshhold, FOLD, TEST_SIZE, Y_col, mixed, date_type, col_remove, y_col_is_null, uids, drop, option
    except Exception as e:
        print(e)
        return ''


if __name__ == '__main__':
    _, parent_path = get_path()
    config_file_name = sys.argv[1]
    print('config_file_name', config_file_name)

    filename, null_threshhold, FOLD, TEST_SIZE, Y_col, mixed, date_type, col_remove, y_col_is_null, uids, drop, option = read_config(config_file_name)
    if len(date_type) == 0:
        df = pd.read_csv(rf'{parent_path}\data\metro\{filename}')
    else:
        df = pd.read_csv(rf'{parent_path}\data\metro\{filename}', parse_dates=date_type)
    if len(uids) == 0:
        pass
    else:
        df = df.set_index(uids, drop=False)
    if len(col_remove) == 0:
        pass
    else:
        df = df.drop(col_remove, axis=1)

    labeler = LabelEncoder()
    df[Y_col] = labeler.fit_transform(df[Y_col])

    cols = df.columns.to_list()
    null_threshhold_cols = []
    no_null_cols = []

    for col in cols:
        null_mean = df[col].isnull().mean()
        if null_mean >= null_threshhold:
            null_threshhold_cols.append(col)
        if null_mean == 0:
            no_null_cols.append(col)

    cols_stayed = [item for item in cols if item not in null_threshhold_cols]
    data = df[cols_stayed].copy()

    # numerical: discrete vs continuous
    date_time = date_type.copy()
    discrete = [var for var in cols_stayed if
                data[var].dtype != 'O' and var != Y_col and var not in date_time and data[var].nunique() < 10]
    continuous = [var for var in cols_stayed if
                  data[var].dtype != 'O' and var != Y_col and var not in date_time and var not in discrete]

    # categorical
    if len(mixed) == 0:
        categorical = [var for var in cols_stayed if data[var].dtype == 'O' and var != Y_col]
    else:
        categorical = [var for var in cols_stayed if data[var].dtype == 'O' and var != Y_col and var not in mixed]

    print('There are {} date_time variables'.format(len(date_time)))
    print('There are {} discrete variables'.format(len(discrete)))
    print('There are {} continuous variables'.format(len(continuous)))
    print('There are {} categorical variables'.format(len(categorical)))

    if y_col_is_null:
        data_notnull = data[data[Y_col] != data[Y_col].max()].copy()
    else:
        data_notnull = data.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        data_notnull.drop(Y_col, axis=1),
        data_notnull[Y_col],
        test_size=TEST_SIZE,
        random_state=0)

    numberImputer = [item for item in continuous + discrete if item not in no_null_cols]
    categoricalImputer = [item for item in categorical if item not in no_null_cols]

    if (len(numberImputer)>0) & (len(categoricalImputer)>0):
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
        if (len(numberImputer)>0) & (len(categoricalImputer)==0):
            pipe = Pipeline([
                ("median_imputer",
                 mdi.MeanMedianImputer(
                     imputation_method="median", variables=numberImputer),)
            ])
        else:
            if (len(numberImputer)==0) & (len(categoricalImputer)>0):
                pipe = Pipeline([
                    ('imputer_cat',
                     mdi.CategoricalImputer(variables=categoricalImputer)),

                    ('categorical_encoder',
                     ce.OrdinalEncoder(encoding_method='ordered',
                                       variables=categoricalImputer))
                ])
            else:
                pipe = []

    X_train, X_test, y_train, y_test = do_imputation(X_train, X_test, y_train, y_test, pipe)
    do_train(X_train, X_test, y_train, y_test,'logic')


    # scaler = MinMaxScaler()  # .set_output(transform="pandas")
    # scaler.fit(X_train_t)
    # X_train_scaled = scaler.transform(X_train_t)
    # X_test_scaled = scaler.transform(X_test_t)
    # write_processed(pd.DataFrame(data=X_train_scaled, columns=X_train_t.columns), y_train, 'X_train_scaled', drop)
    # write_processed(pd.DataFrame(data=X_test_scaled, columns=X_test_t.columns), y_test, 'X_test_scaled', drop)
    # model.fit(X_train_scaled, y_train)
    # y_pred = model.predict(X_test_scaled)
    # print('f1 score with scaling', f1_score(y_test, y_pred, average='macro'))
    #
    # trimmer = ol.OutlierTrimmer(
    #     variables=continuous,
    #     capping_method="iqr",
    #     tail="both",
    #     fold=FOLD,
    # )
    #
    # trimmer.fit(X_train_t)
    # X_train_enc = trimmer.transform(X_train_t)
    # y_train_enc = y_train[X_train_enc.index.to_list()]
    # X_test_enc = trimmer.transform(X_test_t)
    # y_test_enc = y_test[X_test_enc.index.to_list()]
    # model.fit(X_train_enc, y_train_enc)
    # y_pred = model.predict(X_test_enc)
    # print('f1 score with outlier removed', f1_score(y_test_enc, y_pred, average='macro'))
    #
    # write_processed(X_train_enc, y_train_enc, 'X_train_outliered', drop)
    # write_processed(X_test_enc, y_test_enc, 'X_test_outliered', drop)
