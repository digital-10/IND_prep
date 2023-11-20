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
import mean_median2 as mm
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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=configs['y_col'][0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test


def model_selection(option='logic'):
    if option == 'light':
        return lgb.LGBMClassifier(random_state=0)
    else:
        return LogisticRegression(random_state=0)


def read_data(configs):
    if configs['date_col'][0] is np.nan:
        df = pd.read_csv(configs['file_name'][0])
    else:
        df = pd.read_csv(configs['file_name'][0], parse_dates=configs['date_col'])

    if configs['remove_col'][0] == ' ':
        pass
    else:
        remove_cols = configs['remove_col'][0].split(',')
        for rc in remove_cols:
            if rc in df.columns.to_list():
                df = df.drop(rc, axis=1)

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
    if configs['date_col'][0] is np.nan:
        date_exist = False
        date_time = []
    else:
        date_exist = True
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
    if date_exist:
        discrete = [var for var in cols_stayed if
                    data[var].dtype != 'O' and var != Y_col and var not in date_time and data[var].nunique() < 10]
        continuous = [var for var in cols_stayed if
                      data[var].dtype != 'O' and var != Y_col and var not in date_time and var not in discrete]
    else:
        discrete = [var for var in cols_stayed if
                    data[var].dtype != 'O' and var != Y_col and data[var].nunique() < 10]
        continuous = [var for var in cols_stayed if
                      data[var].dtype != 'O' and var != Y_col and var not in discrete]

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


def make_train_test(df, configs):
    df = df.copy()
    X = df.drop(columns=configs['y_col'][0])
    y = df[configs['y_col'][0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=configs['test_size'][0], random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test


def make_imputer_pipe(continuous, discrete, categorical, null_impute_type=None):
    numberImputer = continuous + discrete
    categoricalImputer = categorical
    
    if null_impute_type is None:
        pipe = []
    else:
        if (len(numberImputer) > 0) & (len(categoricalImputer) > 0):
            pipe = Pipeline([
                ("imputer",
                 mm.MeanMedianImputer2(
                     imputation_method=null_impute_type, variables=numberImputer),),
    
                ('imputer_cat',
                 mdi.CategoricalImputer(variables=categoricalImputer)),
    
                ('categorical_encoder',
                 ce.OrdinalEncoder(encoding_method='ordered',
                                   variables=categoricalImputer))
            ])
        else:
            if (len(numberImputer) > 0) & (len(categoricalImputer) == 0):
                pipe = Pipeline([
                    ("imputer",
                     mm.MeanMedianImputer2(
                         imputation_method=null_impute_type, variables=numberImputer),)
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


def do_imputation(df, configs, pipe):
    if pipe != []:
        df = df.copy()
        xtrain, xtest, y_train, y_test = make_train_test(df, configs)
        
        # pipe.fit(X_train, y_train)
        pipe.fit(xtrain, y_train)
        
        X_train = pipe.transform(xtrain)
        X_test = pipe.transform(xtest)

        X_train[configs['y_col'][0]] = y_train        
        X_train['split'] = 'train'
        X_test[configs['y_col'][0]] = y_test
        X_test['split'] = 'test'        
        return pd.concat([X_train, X_test]).reset_index(drop=True)
    else:
        print('no pipe applied')
        return df    


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
    from pathlib import Path

    try:
        folder_name = sys.argv[1]
        config_file_name = sys.argv[2]
        cur_path = os.getcwd()
        parent = os.path.abspath(os.path.join(cur_path, os.pardir))
        config_file = os.path.join(parent, os.path.join('config', f'{config_file_name}'))
        configs = pd.read_excel(config_file, header=None).set_index(0).T
        configs = configs.to_dict('list')
        ori_file_name = configs['file_name'][0].split('.')[0]
        configs['file_name'][0] = os.path.join(parent, os.path.join(f'data/{folder_name}', configs['file_name'][0]))
        print(configs['file_name'][0])
        df_initial = read_data(configs)
    
        # 전처리 저장 경로 정의
        dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
        dest_path = os.path.join(parent, os.path.join(dest_path, 'imputed'))
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        dest_path = os.path.join(parent, os.path.join(dest_path, f'draft_{ori_file_name}.csv'))

        # 오리지널 데이터셋 저장
        df_initial.to_csv(dest_path, index=False)        

        # 1. Label 칼럼 인코딩   
        df, y_null = y_label_enc(df_initial, configs)
        
        # 2. discrete, continuous, categorical 구분작업
        df_organized, discrete, continuous, categorical = organize_data(df, configs, y_null)
                
        # null_impute_types 정의
        null_impute_types = ['median', 'mean', 'max', 'min']
        
        for null_impute_type in null_impute_types:        
            # 3. pipe 작업
            pipe = make_imputer_pipe(discrete, continuous, categorical, null_impute_type=null_impute_type)
            
            # 4. imputation with train/test split
            df_imputed = do_imputation(df_organized, configs, pipe)            
            
            # 5. 전처리 셋 저장    
            dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
            dest_path = os.path.join(parent, os.path.join(f'{dest_path}/imputed', f'imputed_{ori_file_name}_{null_impute_type}.csv'))
            df_imputed.to_csv(dest_path, index=False)
            
            # 6. 스케일링 작업 및 저장
            Y_COL = configs['y_col'][0]
            # 6.1 X_train 스케일링
            con = df_imputed['split'] == 'train'
            X_train_scaled = min_max_scale(df_imputed[con].drop(columns=[Y_COL,'split']))
            X_train_scaled = pd.DataFrame(X_train_scaled)
            X_train_scaled[Y_COL] = df_imputed[con][Y_COL]
            X_train_scaled['split'] = df_imputed[con]['split']
            X_train_scaled.columns = df_imputed.columns

            # 6.2 X_test 스케일링
            con = df_imputed['split'] == 'test'
            X_test_scaled = min_max_scale(df_imputed[con].drop(columns=[Y_COL,'split']))
            X_test_scaled = pd.DataFrame(X_test_scaled)
            tmp = df_imputed.copy().reset_index()
            X_test_scaled['index'] = tmp[con]['index'].values
            X_test_scaled = X_test_scaled.set_index('index')
            X_test_scaled[Y_COL] = df_imputed[con][Y_COL]
            X_test_scaled['split'] = df_imputed[con]['split']
            X_test_scaled.columns = df_imputed.columns
            X_test_scaled.index.name = None
            del tmp

            # 6.3 data frame merge
            df_scaled = pd.concat([X_train_scaled, X_test_scaled])
            # 6.4 scaling 저장
            dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
            dest_path = os.path.join(parent, os.path.join(dest_path, 'scaled'))
            Path(dest_path).mkdir(parents=True, exist_ok=True)
            dest_path = os.path.join(parent, os.path.join(dest_path, f'scaled_{ori_file_name}_{null_impute_type}.csv'))

            df_scaled.to_csv(dest_path, index=False)
    
        print('Completed.')

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('비정상종료', e)
        print(exc_type, exc_tb.tb_lineno)

