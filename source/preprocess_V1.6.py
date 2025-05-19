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

def position_Y_col(cols): # Y label을 가장 뒤로 위치 변경
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
    


