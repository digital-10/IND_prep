import sys
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from feature_engine import imputation as mdi
from feature_engine import encoding as ce
from feature_engine.discretisation import EqualFrequencyDiscretiser
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.encoding import OneHotEncoder
import mean_median2 as mm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def join_abs_path(p1, p2):
    return os.path.abspath(os.path.join(p1, p2))


def position_Y_COL(cols):
    cols = cols.copy()
    cols.remove(Y_COL)
    return cols + [Y_COL]
    

def read_data(afile):    
    if config_dict['date_col'] is np.nan:
        df = pd.read_csv(afile, usecols=config_dict['keep_col'])
    else:
        if config_dict['date_col'] in config_dict['keep_col']:
            df = pd.read_csv(afile, usecols=config_dict['keep_col'], parse_dates=config_dict['date_col'])
        else:
            df = pd.read_csv(afile, usecols=config_dict['keep_col'])
    
    cols = list(df.columns)
    cols = position_Y_COL(cols)
    return df[cols]  # Y lable을 멘뒤로 위치


def y_label_enc(df):
    df = df.copy()
    if df[Y_COL].isnull().any():
        Y_null_exist = True
    else:
        Y_null_exist = False
    labeler = LabelEncoder()
    df[Y_COL] = labeler.fit_transform(df[Y_COL])
    return df, Y_null_exist


def discrete_cont(df):
    data = df.copy()
    # numerical: discrete vs continuous
    if (config_dict['date_col'] is np.nan):
        date_cols_len = 0
    else:
        date_cols_len = len(config_dict['date_col'])
    if date_cols_len < 1:
        discrete = [var for var in data.columns if
                    data[var].dtype != 'O' and var != Y_COL and data[var].nunique() < config_dict['discrete_thresh_hold']]
        continuous = [var for var in data.columns if
                      data[var].dtype != 'O' and var != Y_COL and var not in discrete]
    else:
        discrete = [var for var in data.columns if
                    data[var].dtype != 'O' and var != Y_COL and var not in config_dict['date_col'] and data[var].nunique() < config_dict['discrete_thresh_hold']]
        continuous = [var for var in data.columns if
                      data[var].dtype != 'O' and var != Y_COL and var not in config_dict['date_col'] and var not in discrete]
    
    # categorical
    categorical = [var for var in data.columns if data[var].dtype == 'O' and var != Y_COL]

    print('There are {} date_time variables'.format(date_cols_len))
    print('There are {} discrete variables'.format(len(discrete)))
    print('There are {} continuous variables'.format(len(continuous)))
    print('There are {} categorical variables'.format(len(categorical)))
    
    return discrete, continuous, categorical


def separate_mixed(df):
    df = df.copy()    
    s = config_dict['mixed_str'][0]
    e = config_dict['mixed_str'][1]
    mixed_col = config_dict['mixed'][0]
    df[mixed_col+'num'] = df[mixed_col].str.extract('(\d+)') # captures numerical part
    df[mixed_col+'num'] = df[mixed_col+'num'].astype('float')
    df[mixed_col+'cat'] = df[mixed_col].str[s:e] # captures the first letter
    
    # drop original mixed
    df.drop([mixed_col], axis=1, inplace=True)
    cols = position_Y_COL(list(df.columns))
    return df[cols]


def discretiser(df, numeric):
    df = df.copy()
    method = config_dict['discretiser'][0]
    col = config_dict['discretiser'][1]
    if method == 'equalwidth':
        trans = EqualWidthDiscretiser()
        X = df[[col]]
        trans.fit(X)
        df[col] = trans.transform(X)[col]
    elif method == 'equalfrequency':
        trans = EqualFrequencyDiscretiser()
        X = df[[col]]
        trans.fit(X)
        df[col] = trans.transform(X)[col]
    else:
        print('Method Not Available')
    
    return df


def ohe(df):
    df = df.copy()   
    cols = config_dict['ohe']
    for col in cols:
        trans = OneHotEncoder()
        X = df[[col]]
        trans.fit(X)
        df[col] = trans.transform(X)[col]
    
    return df


def find_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary


def outlier(df):
    df = df.copy()
    cols = config_dict['outlier']
    for c in cols:
        upper_limit, lower_limit = find_boundaries(df, c, config_dict['iqr'])
        outliers_ = np.where(df[c] > upper_limit, True,
                    np.where(df[c] < lower_limit, True, False))
        df = df.loc[~(outliers_)]
    return df    

    
def organize_data(df, y_null_exist):
    df = df.copy()
    cols = list(df.columns)
    cols.remove(Y_COL)
    null_threshhold_cols = []
    
    discrete, continuous, categorical = discrete_cont(df)

    for col in cols:
        null_mean = df[col].isnull().mean()
        if null_mean >= config_dict['null_threshhold']:
            null_threshhold_cols.append(col)

    cols_stayed = [c for c in cols if c not in null_threshhold_cols]
    df = df[cols_stayed+[Y_COL]].copy()

    if y_null_exist:
        df = df[df[Y_COL] != df[Y_COL].max()].copy()

    return df, discrete, continuous, categorical


def make_train_test(df):
    df = df.copy()
    X = df.drop(columns=Y_COL)
    y = df[Y_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config_dict['test_size'], random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test


def make_imputer_pipe(continuous, discrete, categorical, null_impute_type):
    numberImputer = continuous + discrete
    categoricalImputer = categorical.copy()
    categoricalImputer = [item for item in categoricalImputer if item not in config_dict['ohe']]
    oheImputer = config_dict['ohe']

    if (len(numberImputer) > 0) & (len(categoricalImputer) > 0):
        pipe = Pipeline([
            ("imputer",
             mm.MeanMedianImputer2(
                 imputation_method=null_impute_type, variables=numberImputer),),

            ('imputer_cat',
             mdi.CategoricalImputer(variables=categorical)),

            ('categorical_encoder',
             ce.OneHotEncoder(variables=oheImputer)),
            
            ('categorical_encoder2',
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
                     ce.OneHotEncoder(variables=oheImputer)),

                    ('categorical_encoder2',
                     ce.OrdinalEncoder(encoding_method='ordered',
                                       variables=categoricalImputer))
                ])
            else:
                pipe = []
    return pipe


def do_imputation(df, pipe):
    xtrain, xtest, y_train, y_test = make_train_test(df)

    # pipe.fit(X_train, y_train)
    pipe.fit(xtrain, y_train)

    X_train = pipe.transform(xtrain)
    X_test = pipe.transform(xtest)

    X_train[Y_COL] = y_train        
    X_train['split'] = 'train'
    X_test[Y_COL] = y_test
    X_test['split'] = 'test'        
    return pd.concat([X_train, X_test]).reset_index(drop=True)


def scaling(df):    
    df = df.copy()
    if config_dict['scale'] is np.nan:
        config_dict['scale'] =='minmax'   # default with minmax scaling
    if config_dict['scale'] =='minmax':
        scaler = MinMaxScaler()
        scaler.fit(df)
        return scaler.transform(df)
    elif config_dict['scale'] =='standard':
        scaler = StandardScaler()
        scaler.fit(df)
        return scaler.transform(df)
    else: 
        scaler = MinMaxScaler()
        scaler.fit(df)
        return scaler.transform(df)        


if __name__ == '__main__':
    # arv 예1: credit 
    # arv 예2: metro 

    try:
        parent = join_abs_path(os.getcwd(), os.pardir)
        folder = sys.argv[1]    # take input with argv parameter
        conf_file = f'argumet_{folder}.xlsx'      
        
        configs = pd.read_excel(join_abs_path(f'{parent}/config', conf_file), header=None).set_index(0)        
        config_cols = configs.index.tolist()
        config_dict = {}
        for c in config_cols:
            config_dict[c] = configs.loc[c].values[0]
            if (type(config_dict[c]) == int) or (type(config_dict[c]) == float):
                pass
            else:
                config_dict[c] = configs.loc[c].values[0].split(',')
        ori_file_name = config_dict['file_name'][0].split('.')[0]
        if config_dict['mixed_str'] is np.nan or len(config_dict['mixed_str']) < 1:
            pass
        else:
            config_dict['mixed_str'] = [eval(i) for i in config_dict['mixed_str']]
        
            
        if config_dict['y_col'] is np.nan or len(config_dict['y_col']) != 1:
            print('No Y column exists')
            raise Exception
        
        if config_dict['discrete_thresh_hold'] is np.nan or config_dict['discrete_thresh_hold'] < 0:
            print('discrete_thresh_hold set to default 10')
            config_dict['discrete_thresh_hold'] = 10
        
        Y_COL = config_dict['y_col'][0]
        original_file = join_abs_path(f'{parent}/data/{folder}', config_dict['file_name'][0])
        df_initial = read_data(original_file)
        
        # 1. Lable column Encoding
        df_labeld, y_null_exist = y_label_enc(df_initial)
        
        # 2. discrete, continuous, categorical 구분작업
        df_organized, discrete, continuous, categorical = organize_data(df_labeld, y_null_exist)
        
        # 3. Mixed 칼럼을 숫자형/문자형으로 분리
        if config_dict['mixed'] is not np.nan:
            df = separate_mixed(df_organized)
            discrete, continuous, categorical = discrete_cont(df)
        else:
            df = df_organized.copy()
    
        # null_impute_types 정의
        null_impute_types = config_dict['null_imp']
        
        if null_impute_types is not np.nan:
            for null_impute_type in null_impute_types:
        # 4. pipeline 정의
                pipe = make_imputer_pipe(discrete, continuous, categorical, null_impute_type)

                if pipe == []:
                    print('no pipe applied')
                else:
        # 5. imputation thru pipeline
                    df_piped = do_imputation(df, pipe)
                    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder}'))
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'{dest_path}/imputed'))
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'imputed_{ori_file_name}_{null_impute_type}.csv'))
        # 5.1 imputation 저장
                    df_piped.to_csv(dest_path, index=False)
            
        # 6. discretization
                    if config_dict['discretiser'] is not np.nan:    
                        df_piped = discretiser(df_piped, discrete+continuous)
        
        # 7. Outlier 처리
                    if config_dict['outlier'] is not np.nan:    
                        df_piped = outlier(df_piped)
                        df_piped = df_piped.reset_index(drop=True)
        
        # 8. 스케일링 작업 및 저장/ Train과 Test 를 따로 스케일링
        # 8.1 X_train 스케일링
                    con = df_piped['split'] == 'train'
                    X_train_scaled = scaling(df_piped[con].drop(columns=[Y_COL,'split']))
                    X_train_scaled = pd.DataFrame(X_train_scaled)
                    X_train_scaled[Y_COL] = df_piped[con][Y_COL]
                    X_train_scaled['split'] = df_piped[con]['split']
                    X_train_scaled.columns = df_piped.columns
        
         # 8.2 X_test 스케일링
                    con = df_piped['split'] == 'test'
                    X_test_scaled = scaling(df_piped[con].drop(columns=[Y_COL,'split']))
                    X_test_scaled = pd.DataFrame(X_test_scaled)
                    tmp = df_piped.copy().reset_index()
                    X_test_scaled['index'] = tmp[con]['index'].values
                    X_test_scaled = X_test_scaled.set_index('index')
                    X_test_scaled[Y_COL] = df_piped[con][Y_COL]
                    X_test_scaled['split'] = df_piped[con]['split']
                    X_test_scaled.columns = df_piped.columns
                    X_test_scaled.index.name = None
                    del tmp
        
        # 8.3 data frame merge
                    df_scaled = pd.concat([X_train_scaled, X_test_scaled])
        # 8.4 scaling 저장
                    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder}'))
                    dest_path = os.path.join(parent, os.path.join(dest_path, 'scaled'))
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'scaled_{ori_file_name}_{null_impute_type}.csv'))
                    df_scaled.to_csv(dest_path, index=False)
    
        print('Completed.')

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('비정상종료', e)
        print(exc_type, exc_tb.tb_lineno)

