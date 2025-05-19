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
from feature_engine.discretisation import EqualWidthDiscretiser, EqualFrequencyDiscretiser
from feature_engine.encoding import OneHotEncoder
from dateutil.parser import parse
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

import mean_median2 as mm
import temporal_feature as tf
from pathlib import Path
import warnings
import traceback
import json

warnings.filterwarnings("ignore")

def join_abs_path(p1,p2):
    return os.path.abspath(os.path.join(p1,p2))

def position_Y_COL(cols): # Y label을 가장 뒤로 위치 변경
    if Y_COL in cols:  # Y_COL이 있을 때만 remove 실행
        cols_copy = cols.copy()
        cols_copy.remove(Y_COL)
        return cols_copy + [Y_COL]
    else:  # Y_COL이 없으면 변경없이 리턴
        return cols

# 안전한 날짜 파싱 함수
def safe_parse(date_string):
    try:
        return parse(date_string)
    except (ValueError, TypeError):
        return None  # 또는 다른 오류 처리 방법
    
def read_data(afile):
    #날짜컬럼은 temporal_feature.py에서 처리함
    # #case 1 : 날짜 컬럼이 없는 경우
    # if config_dict['date_col'] is np.nan:
    #     df = pd.read_csv(afile, usecols=config_dict['keep_col'])
    # # Case 2: 날짜 컬럼이 있는 경우
    # else:
    #     #date_col이 여러개면 안돌아갈 수 있음
    #     # # Case 2-1: date_col이 keep_col에 포함된 경우
    #     # if config_dict['date_col'] in config_dict['keep_col']:

    #     # Case 2-1: date_col 중 하나라도 keep_col에 포함된 경우
    #     if any(col in config_dict['keep_col'] for col in config_dict['date_col']):
    #         df = pd.read_csv(afile, usecols=config_dict['keep_col'], parse_dates=config_dict['date_col'])
    #         # date_col이 정말 date형인지 확인
    #         for col in config_dict['date_col']:
    #             if not pd.api.types.is_datetime64_any_dtype(df[col]):
    #                 df[col] = df[col].apply(safe_parse)

    #     # Case 2-2: date_col이 keep_col에 없는 경우
    #     else:
    #         df = pd.read_csv(afile, usecols=config_dict['keep_col'])
    df = pd.read_csv(afile, usecols=config_dict['keep_col'])
    cols = list(df.columns)
    cols = position_Y_COL(cols)
    return df[cols]

def y_label_enc(df):
    df = df.copy()
    # 타켓 변수(Y_COL)에 결측치가 있는지 확인
    if df[Y_COL].isnull().any():
        y_null_exist = True
    else:
        y_null_exist = False
    labeler = LabelEncoder()
    #타겟 변수를 숫자로 인코딩
    df[Y_COL] = labeler.fit_transform(df[Y_COL])
    return df, y_null_exist

#분류별 칼럼 분류(discrete : 셀 수 있음, continuous: 연속형, categorical : 오브젝트, 그외 날짜 데이터)
def discrete_cont(df):
    #원본 데이터 보존을 위해 카피하여 작업함
    data = df.copy()
    # 날짜형, 시간형
    date_cols_len = len(config_dict['date_col']) if config_dict['date_col'] and not pd.isna(config_dict['date_col'][0]) else 0

    # json형
    dict_cols_len = len(config_dict['dict_col']) if config_dict['dict_col'] and not pd.isna(config_dict['dict_col'][0]) else 0
    # 벡터형
    vector_cols_len = len(config_dict['vector_col']) if config_dict['vector_col'] and not pd.isna(config_dict['vector_col'][0]) else 0
    # 진법형
    non_dec_cols_len = len(config_dict['non_dec_col']) if config_dict['non_dec_col'] and not pd.isna(config_dict['non_dec_col'][0]) else 0
    # 문장형
    sentence_cols_len = len(config_dict['sentence_col']) if config_dict['sentence_col'] and not pd.isna(config_dict['sentence_col'][0]) else 0
    
    # Case 1 : 날짜 컬럼이 없으면
    if date_cols_len < 1:
        # 이산형 변수: 숫자형이면서 고유값이 임계값보다 적은 경우
        discrete = [var for var in data.columns if
                    data[var].dtype != 'O' and var != Y_COL and data[var].nunique() < config_dict['discrete_thresh_hold']]
        # 연속형 변수: 숫자형이면서 이산형이 아닌 경우
        continuous = [var for var in data.columns if
                      data[var].dtype != 'O' and var != Y_COL and var not in discrete]
    # Case 2 : 날짜 컬럼이 있으면
    else: 
        # 이산형 변수 : 숫자형이면서 고유값이 임계값보다 적은 경우 및 날자 컬럼이 아닌 경우
        discrete = [var for var in data.columns if
                    data[var].dtype != 'O' and var != Y_COL and var not in config_dict['date_col']
                    and var not in config_dict['dict_col'] and var not in config_dict['vector_col']
                    and var not in config_dict['non_dec_col'] and var not in config_dict['sentence_col']
                    and data[var].nunique() < config_dict['discrete_thresh_hold']]
        # 연속형 변수 : 숫자형이면서 이산형이 아닌 경우 및 날자컬럼이 아닌 경우
        continuous = [var for var in data.columns if
                      data[var].dtype != 'O' and var != Y_COL and var not in config_dict['date_col']
                      and var not in config_dict['dict_col'] and var not in config_dict['vector_col']
                      and var not in config_dict['non_dec_col'] and var not in config_dict['sentence_col']
                      and var not in discrete]
    # categorical
    # 객체형(문자열) 데이터이면서 타겟변수가 아닌 경우
    categorical = [var for var in data.columns if
                data[var].dtype == 'O' and var != Y_COL and var not in config_dict['date_col']
                and var not in config_dict['dict_col'] and var not in config_dict['vector_col']
                and var not in config_dict['non_dec_col'] and var not in config_dict['sentence_col']]

    # 전처리 데이터 타입 확인용
    print(f'There are {date_cols_len} date_time variables')
    print(f'There are {dict_cols_len} dict variables')
    print(f'There are {vector_cols_len} vector variables')
    print(f'There are {non_dec_cols_len} non-decimal variables')
    print(f'There are {sentence_cols_len} sentence variables')
    print(f'There are {len(discrete)} discrete variables')
    print(f'There are {len(continuous)} continuous variables')
    print(f'There are {len(categorical)} categorical variables')
    return discrete, continuous, categorical
     
def separate_mixed(df):
    df = df.copy()
    s = config_dict['mixed_str'][0]
    e = config_dict['mixed_str'][1]
    mixed_col = config_dict['mixed'][0]
    df[mixed_col+'num'] = df[mixed_col].str.extract(r'(\d+)') #captures numerical part
    df[mixed_col+'num'] = df[mixed_col+'num'].astype('float')
    df[mixed_col+'cat'] = df[mixed_col].str[s:e] # captures the first letter

    # drop original mixed
    df.drop([mixed_col], axis = 1, inplace=True)
    cols = position_Y_COL(list(df.columns))
    return df[cols]

#소수형을 정수형으로 
def truncate_to_integer(series):
    # 모든 값이 1보다 클 때까지 10을 곱함
    while (series < 1).any():
        series *= 10
    # 소수점 이하 잘라내고 정수로 변환
    truncated_series = series.astype(int)

    #원본 값과 변환된 값의 관계 저장(XAI 필요하면 사용)
    #value_map = pd.Series(truncated_series.values, index=series.values)

    return truncated_series

# 정수형의 1의 자리를 버림
def truncate_to_ten(series):
    series /= 10
    # 소수점 이하 잘라내고 정수로 변환
    truncated_series = series.astype(int)
    truncated_series *= 10
    # 원본 값과 변환된 값의 관계 저장(XAI 필요하면 사용)
    #value_map = pd.Series(truncated_series.values, index=series.values)

    return truncated_series

def discretiser(df, numeric):
    df = df.copy()
    method = config_dict['discretiser_type'][0]
    cols = config_dict['discretiser']
    for col in cols:
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
        elif method == 'equalfixed':
            
            # 실수형이면 truncate_to_integer 함수 호출
            if np.issubdtype(df[col].dtype, np.floating):  # 실수형 확인
                truncated_data = truncate_to_integer(df[col])
                df[col] = truncated_data  # 변환된 정수형 데이터로 대체
            # 정수형이면 truncate_to_ten 함수 호출
            else:
                truncated_data = truncate_to_ten(df[col])
                df[col] = truncated_data  # 변환된 정수형 데이터로 대체
        else:
            print(f'Method Not Available for column {col}')
        
