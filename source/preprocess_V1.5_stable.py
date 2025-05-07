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
from dateutil.parser import parse
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
#from feature_engine.imputation import MeanMedianImputer
#from feature_engine.encoding import OneHotEncoder, OrdinalEncoder
#from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser

import mean_median2 as mm
import temporal_feature as tf
from pathlib import Path
import warnings
import traceback
import json

warnings.filterwarnings('ignore')


def join_abs_path(p1, p2):
    return os.path.abspath(os.path.join(p1, p2))


def position_Y_COL(cols):  # Y label을 가장 뒤로 위치 변경
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
    # # Case 1: 날짜 컬럼이 없는 경우
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
    
    df = pd.read_csv(afile, usecols=config_dict['keep_col'], encoding='utf-8')
    cols = list(df.columns)
    cols = position_Y_COL(cols)
    return df[cols]  


def y_label_enc(df):
    df = df.copy()
    # 타겟 변수(Y_COL)에 결측치가 있는지 확인
    if df[Y_COL].isnull().any():
        Y_null_exist = True
    else:
        Y_null_exist = False
    labeler = LabelEncoder()
    # 타겟 변수를 숫자로 인코딩
    df[Y_COL] = labeler.fit_transform(df[Y_COL])
    return df, Y_null_exist

#분류별 컬럼 분류(discrete:셀수있음, continuous:연속형, categorical:오브젝트, 그외 날짜 데이터)
def discrete_cont(df):
    # 원본 데이터 보존을 위해 카피하여 작업함
    data = df.copy()
    # 날짜형, 시간형
    if (config_dict['date_col'] is np.nan):
        date_cols_len = 0
    else:
        date_cols_len = len(config_dict['date_col'])
    
    # json형
    if (config_dict['dict_col'] is np.nan):
        dict_cols_len = 0
    else:
        dict_cols_len = len(config_dict['dict_col'])
    
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
        # 이산형 변수: 숫자형이면서 고유값이 임계값보다 적은 경우 및 날자컬럼이 아닌 경우
        discrete = [var for var in data.columns if
                    data[var].dtype != 'O' and var != Y_COL and var not in config_dict['date_col'] and var not in config_dict['dict_col'] and data[var].nunique() < config_dict['discrete_thresh_hold']]
        # 연속형 변수: 숫자형이면서 이산형이 아닌 경우 및 날자컬럼이 아닌 경우
        continuous = [var for var in data.columns if
                      data[var].dtype != 'O' and var != Y_COL and var not in config_dict['date_col'] and var not in config_dict['dict_col'] and var not in discrete]

    # categorical
    # 객체형(문자열) 데이터이면서 타겟변수가 아닌 경우
    categorical = [var for var in data.columns if data[var].dtype == 'O' and var != Y_COL and var not in config_dict['date_col'] and var not in config_dict['dict_col']]
    
    # 전처리 데이터 타입 확인용
    print('There are {} date_time variables'.format(date_cols_len))
    print('There are {} dict variables'.format(dict_cols_len))
    print('There are {} discrete variables'.format(len(discrete)))
    print('There are {} continuous variables'.format(len(continuous)))
    print('There are {} categorical variables'.format(len(categorical)))
    return discrete, continuous, categorical


def separate_mixed(df):
    df = df.copy()    
    s = config_dict['mixed_str'][0]
    e = config_dict['mixed_str'][1]
    mixed_col = config_dict['mixed'][0]
    df[mixed_col+'num'] = df[mixed_col].str.extract(r'(\d+)') # captures numerical part
    df[mixed_col+'num'] = df[mixed_col+'num'].astype('float')
    df[mixed_col+'cat'] = df[mixed_col].str[s:e] # captures the first letter

    # drop original mixed
    df.drop([mixed_col], axis=1, inplace=True)
    cols = position_Y_COL(list(df.columns))
    return df[cols]

# 소수형을 정수형으로
def truncate_to_integer(series):
    # 모든 값이 1보다 클 때까지 10을 곱함
    while (series < 1).any():
        series *= 10
    
    # 소수점 이하 잘라내고 정수로 변환
    truncated_series = series.astype(int)
    
    # 원본 값과 변환된 값의 관계 저장(XAI 필요하면 사용)
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
    for col in cols:  # 각 열에 대해 반복
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

        # 실수형이면 truncate_to_integer 함수 호출
        if np.issubdtype(df[col].dtype, np.floating):  # 실수형 확인
            truncated_data = truncate_to_integer(df[col])
            df[col] = truncated_data  # 변환된 정수형 데이터로 대체

    return df

#안씀 - make_imputer_pipe 로 대체
def ohe(df):
    df = df.copy()   
    cols = config_dict['ohe']
    for col in cols:
        trans = OneHotEncoder()
        X = df[[col]]
        trans.fit(X)
        df[col] = trans.transform(X)[col]
    return df

#이상치 탐색을 위한 평균값 구하기
def find_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

#이상치 제거 함수
def outlier(df):
    df = df.copy()
    cols = config_dict['outlier']
    for c in cols:
        upper_limit, lower_limit = find_boundaries(df, c, config_dict['iqr'])
        outliers_ = np.where(df[c] > upper_limit, True,
                    np.where(df[c] < lower_limit, True, False))
        df = df.loc[~(outliers_)]
    return df    

#json 데이터 처리
def extract_json_data(df):
    """
    주어진 JSON 문자열 데이터를 파싱하여 딕셔너리 형태로 변환하는 함수

    Parameters:
    json_column_data (list): JSON 문자열이 포함된 리스트

    Returns:
    DataFrame: JSON 데이터가 포함된 새로운 DataFrame
    """
    cols = config_dict['dict_col']
    df = df.copy()
    for col in cols:
        json_column_data = df[col].tolist()
        # JSON 데이터를 저장할 리스트
        json_records = []

        for json_str in json_column_data:
            try:
                # JSON 문자열을 파싱
                json_data = json.loads(json_str)
                json_records.append(json_data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"JSON 파싱 오류: {str(e)}")
                json_records.append({})  # 오류 발생 시 빈 딕셔너리 추가
        
        # JSON 데이터를 DataFrame으로 변환
        json_df = pd.DataFrame(json_records)

        # 새로운 컬럼명 생성: 기존 컬럼명 + "_" + JSON 키
        new_column_names = {key: f"{col}_{key}" for key in json_df.columns}
        
        # 새로운 컬럼명으로 DataFrame의 컬럼명 변경
        json_df.rename(columns=new_column_names, inplace=True)
        df = pd.concat([df, json_df], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df


def parse_vector_string(vec_str):
    """문자열 형태의 벡터를 파싱하여 numpy 배열로 변환"""
    try:
        # '[1.2, 3.4]' 같은 문자열을 실제 리스트로 변환 후 numpy 배열로
        return np.array(ast.literal_eval(vec_str))
    except (ValueError, SyntaxError, TypeError):
        # 파싱 실패 시 None 또는 적절한 값 반환 (예: np.nan 배열)
        # 차원 수가 맞지 않을 수 있으므로 주의 필요
        return None # 또는 np.array([np.nan] * expected_dimension)

def apply_pca(df):
    """DataFrame에 PCA를 적용하는 함수"""
    df = df.copy()
    if 'pca_col' not in config_dict or config_dict['pca_col'] is np.nan:
        return df # 설정 없으면 원본 반환

    pca_cols = config_dict['pca_col']
    n_components = int(config_dict['pca_n_components']) # 설정 파일에서 개수 읽기 (첫번째 값 사용)

    for col in pca_cols:
        if col not in df.columns:
            print(f"Warning: PCA target column '{col}' not found in DataFrame. Skipping.")
            continue

        # 1. 벡터 문자열 파싱 (문자열 형태일 경우)
        # 실제 데이터 타입에 따라 parse_vector_string 또는 다른 전처리 필요
        parsed_vectors = df[col].apply(parse_vector_string)

        # 파싱 실패한 경우 제외 (또는 다른 방식으로 처리)
        valid_vectors_idx = parsed_vectors.notna()
        if not valid_vectors_idx.any():
            print(f"Warning: No valid vectors found in column '{col}' after parsing. Skipping PCA.")
            continue

        # 유효한 벡터들을 numpy 배열로 스택
        vector_matrix = np.stack(parsed_vectors[valid_vectors_idx].values)

        # 2. 스케일링 (PCA는 스케일에 민감)
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(vector_matrix)

        # 3. PCA 적용
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_matrix)

        # 4. 결과를 새 컬럼으로 DataFrame에 추가
        pc_cols = [f"{col}_PC{i+1}" for i in range(n_components)]
        pc_df = pd.DataFrame(principal_components, index=df[valid_vectors_idx].index, columns=pc_cols)

        # 원본 df와 병합 (NaN 처리 주의)
        df = pd.concat([df, pc_df], axis=1)

        # 5. 원본 벡터 컬럼 제거 (선택 사항)
        df.drop(col, axis=1, inplace=True)
        print(f"Applied PCA on '{col}', created columns: {pc_cols}")

    return df


def convert_base_to_int_value(value):
    """단일 값을 10진수 정수로 변환"""
    if not isinstance(value, str): # 문자열이 아니면 변환 시도 안 함
        return value # 또는 오류 처리
    value = value.lower().strip() # 소문자 변환 및 공백 제거
    try:
        if value.startswith('0x'):
            return int(value, 16)
        elif value.startswith('0b'):
            return int(value, 2)
        elif value.startswith('0o'):
            return int(value, 8)
        else:
            # 접두어 없으면 10진수로 간주 (또는 오류/None 반환)
            return int(value, 10)
    except (ValueError, TypeError):
        # 변환 실패 시 원래 값 또는 np.nan 반환
        return np.nan # 또는 원래 value 반환

def apply_base_conversion(df):
    """DataFrame에 진법 변환 적용"""
    df = df.copy()
    if 'base_conv_col' not in config_dict or config_dict['base_conv_col'] is np.nan:
        return df

    base_conv_cols = config_dict['base_conv_col']

    for col in base_conv_cols:
        if col not in df.columns:
            print(f"Warning: Base conversion target column '{col}' not found. Skipping.")
            continue

        df[col] = df[col].apply(convert_base_to_int_value)
        # 변환 후 데이터 타입을 정수 또는 실수(NaN 포함 시)로 변경 시도
        df[col] = pd.to_numeric(df[col], errors='ignore')
        print(f"Applied base conversion on column '{col}'.")

    return df

# 모델 로드는 시간이 걸리므로 함수 밖 또는 클래스 초기화 시 한 번만 수행하는 것이 효율적
# 여기서는 간단히 함수 내에서 로드 (필요 시 수정)
def get_sentence_transformer_model(model_name='paraphrase-MiniLM-L6-v2'):
    """Sentence Transformer 모델 로드"""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error loading sentence transformer model '{model_name}': {e}")
        return None

def apply_sentence_embedding(df):
    """DataFrame에 문장 임베딩 적용"""
    df = df.copy()
    if 'sentence_col' not in config_dict or config_dict['sentence_col'] is np.nan:
        return df

    sentence_cols = config_dict['sentence_col']
    model_name = config_dict.get('sentence_model') # 설정 없으면 기본값
    if pd.isna(model_name) or not model_name:
        model_name = 'paraphrase-MiniLM-L6-v2'

    model = get_sentence_transformer_model(model_name)
    if model is None:
        print("Skipping sentence embedding due to model loading failure.")
        return df

    for col in sentence_cols:
        if col not in df.columns:
            print(f"Warning: Sentence embedding target column '{col}' not found. Skipping.")
            continue

        # 결측치 처리 (예: 빈 문자열로 대체)
        sentences = df[col].fillna('').astype(str).tolist()

        # 문장 임베딩 생성 (시간/메모리 소요)
        print(f"Generating sentence embeddings for column '{col}' using model '{model_name}'...")
        embeddings = model.encode(sentences, show_progress_bar=True)

        # 결과를 새 컬럼들로 DataFrame에 추가
        emb_dim = embeddings.shape[1]
        emb_cols = [f"{col}_emb_{i}" for i in range(emb_dim)]
        emb_df = pd.DataFrame(embeddings, index=df.index, columns=emb_cols)

        # 원본 df와 병합
        df = pd.concat([df, emb_df], axis=1)

        # 원본 문장 컬럼 제거 (선택 사항)
        df.drop(col, axis=1, inplace=True)
        print(f"Applied sentence embedding on '{col}', created {emb_dim} embedding columns.")

    return df



#데이터 전처리 함수
def organize_data(df, y_null_exist):
    df = df.copy()
    cols = list(df.columns)
    cols.remove(Y_COL)
    null_threshhold_cols = []
    discrete, continuous, categorical = discrete_cont(df)
    # - discrete: 이산형 변수
    # - continuous: 연속형 변수
    # - categorical: 범주형 변수

    #널 비율이 임계치를 넘은 컬럼 명세 작성
    for col in cols:
        null_mean = df[col].isnull().mean() # 각 컬럼의 null 비율 계산
        if null_mean >= config_dict['null_threshhold']:
            null_threshhold_cols.append(col)

    #임계치를 넘은 대상을 빼고 다시 df 만듬
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

#각 형태별로 파이프라인을 추가하는 형태로 변경
def make_imputer_pipe_old(continuous, discrete, categorical, null_impute_type):
    # 연속형 변수와 이산형 변수를 합쳐서 수치형 변수로 처리
    numberImputer = continuous + discrete

    categoricalImputer = categorical.copy()
    # One-Hot Encoding 대상 변수 제외
    categoricalImputer = [item for item in categoricalImputer if (item not in config_dict['ohe']) ]
    oheImputer = config_dict['ohe']
    
    result={}
    
    # 수치형 변수와 범주형 변수가 모두 있는 경우
    if (len(numberImputer) > 0) & (len(categoricalImputer) > 0):
        pipe = Pipeline([
            # 수치형 변수 결측치 대체
            ("imputer",
            mm.MeanMedianImputer2(
                imputation_method=null_impute_type, variables=numberImputer),),
            # 범주형 변수 결측치 대체
            ('imputer_cat',
            mdi.CategoricalImputer(variables=categorical)),
            # One-Hot Encoding 적용
            ('categorical_encoder',
            ce.OneHotEncoder(variables=oheImputer)),
            # 라벨링 인코딩 적용
            ('categorical_encoder2',
            ce.OrdinalEncoder(encoding_method='ordered',
                variables=categoricalImputer))
        ])
    else:
        # 수치형 변수만 있고 범주형 변수가 없는 경우
        if (len(numberImputer) > 0) & (len(categoricalImputer) == 0):
            pipe = Pipeline([
                # 수치형 변수 결측치만 대체
                ("imputer",
                mm.MeanMedianImputer2(
                    imputation_method=null_impute_type, variables=numberImputer),)
            ])
        else:
            # 범주형 변수만 있고 수치형 변수가 없는 경우
            if (len(numberImputer) == 0) & (len(categoricalImputer) > 0):
                pipe = Pipeline([
                    # 범주형 변수 결측치 대체
                    ('imputer_cat',
                    mdi.CategoricalImputer(variables=categorical)),
                    ('categorical_encoder',
                    ce.OneHotEncoder(variables=oheImputer)),
                    ('categorical_encoder2',
                    ce.OrdinalEncoder(encoding_method='ordered',
                        variables=categoricalImputer))
                ])
            else:
                pipe = []
    return pipe

def make_imputer_pipe(continuous, discrete, categorical, null_impute_type):
    # 연속형 변수와 이산형 변수를 합쳐서 수치형 변수로 처리
    numberImputer = continuous + discrete

    categoricalImputer = categorical.copy()
    # One-Hot Encoding 대상 변수 제외
    categoricalImputer = [item for item in categoricalImputer if (item not in config_dict['ohe']) ]
    oheImputer = config_dict['ohe']
    datecolImputer = config_dict['date_col']
    
    result={}
    
    steps = []
    # 수치형 변수 처리 파이프라인(결측치를 null_impute_type값[mean,median,max,min]에 따라 채움)
    if numberImputer and len(numberImputer) > 0:
        steps.append(
            ("numeric_imputer", 
            mm.MeanMedianImputer2(
                imputation_method=null_impute_type,
                variables=numberImputer
                )
            )
        )

    # 범주형 변수 처리 파이프라인(결측치를 최빈값으로 채움)
    if categorical and len(categorical) > 0:
        steps.append(
            ('categorical_imputer',
            mdi.CategoricalImputer(variables=categorical))
        )

    # 원핫인코딩 처리(데이터 종류만큼 컬럼을 만들어 1,0으로 표현)
    if oheImputer and len(oheImputer) > 0:
        steps.append(
            ('onehot_encoder',
            ce.OneHotEncoder(variables=oheImputer))
        )

    # 라벨 인코딩 처리(데이터 종류별 고유수치로 변경, male->1, female->2)
    if categoricalImputer and len(categoricalImputer) > 0:
        steps.append(
            ('label_encoder',
            ce.OrdinalEncoder(
                encoding_method='ordered',
                variables=categoricalImputer
                )
            )
        )
    # 시계열 데이터 처리(날짜형에서 연월일 추출, 시간형에서 타임델타 추출)
    if datecolImputer and len(datecolImputer) > 0:
        steps.append(
            ('temporal_feature_engineering',
            tf.DateFeatureTransformer2(
                variables=datecolImputer,
                features=['year', 'month', 'day', 'time_seconds'],
                drop_original=True
                )
            )
        )
    
    # 파이프라인 생성
    if steps:
        return Pipeline(steps)
    return []


def do_imputation(df, pipe):
    train=False
    if(train):
        xtrain, xtest, y_train, y_test = make_train_test(df)
        
        # pipe.fit(X_train, y_train)
        # 파이프라인을 훈련 데이터에 맞춤
        pipe.fit(xtrain, y_train)
        X_train = pipe.transform(xtrain)
        X_test = pipe.transform(xtest)

        # 훈련 세트에 타겟 변수와 'split' 열 추가
        X_train[Y_COL] = y_train        
        X_train['split'] = 'train'
        # 테스트 세트에 타겟 변수와 'split' 열 추가
        X_test[Y_COL] = y_test
        X_test['split'] = 'test'        
        return pd.concat([X_train, X_test]).reset_index(drop=True)
    else:
        # 전체 데이터에 대해 파이프라인을 적용
        # 타겟 변수 분리
        y_full = df[Y_COL]
        
        # 파이프라인을 전체 데이터에 맞춤
        pipe.fit(df.drop(columns=[Y_COL]),y_full)
        
        # 변환 적용
        X_full = pipe.transform(df.drop(columns=[Y_COL]))
        
        # 변환된 데이터프레임에 타겟 변수 추가
        X_full[Y_COL] = y_full
        X_full['split'] = 'full'
        
        return X_full.reset_index(drop=True)

def scaling(df):    
    df = df.copy()
    if config_dict['scale'] is np.nan:
        config_dict['scale'] = ['minmax']   # default with minmax scaling
    if config_dict['scale'][0] =='minmax':
        scaler = MinMaxScaler()
        scaler.fit(df)
        return scaler.transform(df)
    elif config_dict['scale'][0] =='standard':
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
        #파라미터
        #folder = sys.argv[1]    # take input with argv parameter
        folder = "loans"    #테스트용

        parent = join_abs_path(os.getcwd(), os.pardir)
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
        
        #mixed_str의 정수변환
        if config_dict['mixed_str'] is np.nan or len(config_dict['mixed_str']) < 1:
            pass
        else:
            config_dict['mixed_str'] = [eval(i) for i in config_dict['mixed_str']]  #배열의 각 값을 정수형으로 변환

        if config_dict['y_col'] is np.nan or len(config_dict['y_col']) != 1:
            print('No Y column exists')
            raise Exception

        if config_dict['discrete_thresh_hold'] is np.nan or config_dict['discrete_thresh_hold'] < 0:
            print('discrete_thresh_hold set to default 10')
            config_dict['discrete_thresh_hold'] = 10

        Y_COL = config_dict['y_col'][0]
        original_file = join_abs_path(f'{parent}/data/{folder}', config_dict['file_name'][0])
        df_initial = read_data(original_file)
        # --- 진법 변환 추가 ---
        if 'base_conv_col' in config_dict and config_dict['base_conv_col'] is not np.nan:
            df_base_converted = apply_base_conversion(df_initial) # 진법 변환 함수 호출
        else:
            df_base_converted = df_initial

        # --- 문장 임베딩 추가 ---
        if 'sentence_col' in config_dict and config_dict['sentence_col'] is not np.nan:
            df_embedded = apply_sentence_embedding(df_base_converted) # 임베딩 함수 호출
        else:
            df_embedded = df_base_converted

        # 1. Label column Encoding
        df_labeld, y_null_exist = y_label_enc(df_embedded)

        # 1.1json 처리
        df_jsoned = extract_json_data(df_labeld)
        # --- PCA 적용 추가 ---
        if 'pca_col' in config_dict and config_dict['pca_col'] is not np.nan:
            df_pca_applied = apply_pca(df_jsoned) # PCA 적용 함수 호출
        else:
            df_pca_applied = df_jsoned # PCA 설정 없으면 그대로 진행

        # 2. discrete, continuous, categorical 구분작업
        df_organized, discrete, continuous, categorical = organize_data(df_pca_applied, y_null_exist)

        # 3. Mixed 칼럼을 숫자형/문자형으로 분리(분리 후 df_organized, discrete, continuous, categorical 재분류)
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
        # 5. discretization(연속형 변수를 범주형으로)
                    if config_dict['discretiser'] is not np.nan:
                        df_piped = discretiser(df, discrete+continuous)
        # 6. imputation thru pipeline
                    df_piped = do_imputation(df, pipe)
                    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder}'))
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'{dest_path}/imputed'))
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'imputed_{ori_file_name}_{null_impute_type}.csv'))
        # 7.1 imputation 저장
                    df_piped.to_csv(dest_path, index=False)

        # 8. discretization(연속형 변수를 범주형으로)
                    if config_dict['discretiser'] is not np.nan:
                        df_piped = discretiser(df_piped, discrete+continuous)
        
        # 9. Outlier 처리
                    if config_dict['outlier'] is not np.nan:    
                        df_piped = outlier(df_piped)
                        df_piped = df_piped.reset_index(drop=True)
    

        # 9.1 데이터 정제 저장
                    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder}'))
                    dest_path = os.path.join(parent, os.path.join(dest_path, 'trans'))
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'trans_{ori_file_name}_{null_impute_type}.csv'))
                    df_piped.to_csv(dest_path, index=False)


        # 10. 스케일링 작업 및 저장/ Train과 Test 를 따로 스케일링
        # 10.1 X_train 스케일링
                    con = df_piped['split'] == 'train'
                    if not df_piped[con].empty:
                        X_train_scaled = scaling(df_piped[con].drop(columns=[Y_COL,'split']))
                        X_train_scaled = pd.DataFrame(X_train_scaled)
                        X_train_scaled[Y_COL] = df_piped[con][Y_COL]
                        X_train_scaled['split'] = df_piped[con]['split']
                        X_train_scaled.columns = df_piped.columns
                    else:
                        X_train_scaled = []
        # 10.2 X_test 스케일링
                    con = df_piped['split'] == 'test'
                    if not df_piped[con].empty:
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
                    else:
                        X_test_scaled = []
        # 10.3 data frame merge
                    if (len(X_train_scaled) == 0 and len(X_test_scaled) == 0 ):
                        df_scaled = scaling(df_piped.drop(columns=[Y_COL,'split']))
                        df_scaled = pd.DataFrame(df_scaled)
                        df_scaled[Y_COL] = df_piped[Y_COL]
                        df_scaled['split'] = df_piped['split']
                        df_scaled.columns = df_piped.columns
                    else :
                        df_scaled = pd.concat([X_train_scaled, X_test_scaled])
        # 10.4 scaling 저장
                    dest_path = os.path.join(parent, os.path.join('data_preprocessed', f'{folder}'))
                    dest_path = os.path.join(parent, os.path.join(dest_path, 'scaled'))
                    Path(dest_path).mkdir(parents=True, exist_ok=True)
                    dest_path = os.path.join(parent, os.path.join(dest_path, f'scaled_{ori_file_name}_{null_impute_type}.csv'))
                    df_scaled.to_csv(dest_path, index=False)
        print('Completed.')
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('비정상종료', e)
        traceback.print_exc()
        print(exc_type, exc_tb.tb_lineno)