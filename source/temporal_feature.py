"""
Created on 2024-11-18

@author: sjh
"""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import re

class DateFeatureTransformer2(BaseEstimator, TransformerMixin):
    """
    날짜형 데이터를 연, 월, 일, 요일 등의 특성으로 분해하는 transformer
    
    Parameters
    ----------
    variables : list or str
        변환할 날짜 컬럼명들
    features : list, default ['year', 'month', 'day', 'dayofweek']
        추출할 날짜 특성들. 가능한 값들:
        - 'year': 연도
        - 'month': 월
        - 'day': 일
        - 'dayofweek': 요일 (0:월요일 ~ 6:일요일)
        - 'quarter': 분기
        - 'dayofyear': 일년 중 몇번째 일
        - 'weekofyear': 일년 중 몇번째 주
        - 'is_month_start': 월의 시작일 여부
        - 'is_month_end': 월의 마지막일 여부
    drop_original : bool, default True
        원본 날짜 컬럼 삭제 여부
    """
    
    def __init__(self, variables, features=['year', 'month', 'day', 'dayofweek'], 
                drop_original=True):
        self.variables = [variables] if isinstance(variables, str) else variables
        self.features = features
        self.drop_original = drop_original
        
        # 지원하는 날짜 특성들
        self.available_features = {
            'year': lambda x: x.dt.year,
            'month': lambda x: x.dt.month,
            'day': lambda x: x.dt.day,
            'dayofweek': lambda x: x.dt.dayofweek,
            'quarter': lambda x: x.dt.quarter,
            'dayofyear': lambda x: x.dt.dayofyear,
            'weekofyear': lambda x: x.dt.isocalendar().week,
            'is_month_start': lambda x: x.dt.is_month_start.astype(int),
            'is_month_end': lambda x: x.dt.is_month_end.astype(int),
            # 시간을 초단위로 변환 (하루 중 경과 초)
            'time_seconds': lambda x: (x.dt.hour * 3600 + 
                                    x.dt.minute * 60 + 
                                    x.dt.second +
                                    x.dt.microsecond / 1000000),
        }
        
        # 요청된 특성이 지원되는지 확인
        for feature in self.features:
            if feature not in self.available_features:
                raise ValueError(f"'{feature}' is not a supported date feature. "
                                f"Available features: {list(self.available_features.keys())}")
    
    def _is_time_only(self, s):
        """
        문자열이 시간만 포함하고 있는지 확인
        """
        time_only_patterns = [
            r'^\d{1,2}:\d{2}(:\d{2})?(\.\d+)?$',  # HH:MM:SS.fff
            r'^\d{1,2}:\d{2}(:\d{2})?(\s?[AaPp][Mm])?$'  # HH:MM:SS AM/PM
        ]
        return any(re.match(pattern, str(s).strip()) for pattern in time_only_patterns)
    def _get_valid_features(self, value):
        """
        데이터 형식에 따라 적절한 특성 목록 반환
        """
        if pd.isna(value):
            return []
        if self._is_time_only(value):
            # 시간만 있는 경우 시간 관련 특성만 반환
            return [f for f in self.features if f in ['time_seconds']]
        # 전체 날짜가 있는 경우 모든 요청 특성 반환
        return self.features

    def _convert_to_datetime(self, series):
        """
        시리즈를 datetime으로 변환
        시간만 있는 경우 임의의 날짜(1900-01-01)를 사용
        """
        def convert_value(value):
            if pd.isna(value):
                return pd.NaT
            
            value_str = str(value).strip()
            if self._is_time_only(value_str):
                # 시간만 있는 경우 임의의 날짜와 결합
                dummy_date = '1900-01-01 '
                return pd.to_datetime(dummy_date + value_str)
            try:
                # 날짜만 있는 경우
                return pd.to_datetime(value_str).normalize()
            except:
                # 날짜와 시간이 모두 있는 경우
                return pd.to_datetime(value_str)
        
        return series.apply(convert_value)
    
    def fit(self, X, y=None):
        """
        데이터 타입 검증 및 날짜형으로 변환 가능한지 확인
        """
        # 입력 데이터가 DataFrame인지 확인
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        # 지정된 컬럼이 존재하는지 확인
        if not all(var in X.columns for var in self.variables):
            missing_vars = [var for var in self.variables if var not in X.columns]
            raise ValueError(f"Variables {missing_vars} not found in input data")
            
        # 날짜형으로 변환 가능한지 확인
        for var in self.variables:
            try:
                pd.to_datetime(X[var])
            except Exception as e:
                raise ValueError(f"Column {var} cannot be converted to datetime: {str(e)}")
                
        return self
    
    def transform(self, X):
        """
        지정된 날짜 컬럼들에서 특성 추출
        """
        X = X.copy()
        
        for var in self.variables:
            
            # 전체 컬럼에 대한 데이터 형식 확인
            date_series = self._convert_to_datetime(X[var])
            
            # 컬럼의 전반적인 특성 결정
            if (date_series.dt.time == pd.Timestamp('00:00:00').time()).all():
                # 모든 값이 날짜만 있는 경우
                valid_features = [f for f in self.features if f != 'time_seconds']
            elif (date_series.dt.date == pd.Timestamp('1900-01-01').date()).all():
                # 모든 값이 시간만 있는 경우
                valid_features = ['time_seconds']
            else:
                # 날짜와 시간이 모두 있는 경우
                valid_features = self.features

            # 특성 추출
            for feature in valid_features:
                new_column = f"{var}_{feature}"
                if feature == 'time_seconds' and 'time_seconds' in valid_features:
                    # 시간만 있는 경우, 1900-01-01 기준으로 계산
                    X[new_column] = (date_series.dt.hour * 3600 +
                                    date_series.dt.minute * 60 +
                                    date_series.dt.second +
                                    date_series.dt.microsecond / 1000000).astype(float)
                    # 소수점 한 자리로 포맷팅
                    X[new_column] = X[new_column].map(lambda x: format(x, '.1f'))
                else:
                    X[new_column] = self.available_features[feature](date_series)
            
            # 원본 컬럼 삭제 옵션
            if self.drop_original:
                X = X.drop(columns=[var])
        return X
# 사용 예시:
"""
# Transformer 생성
date_transformer = DateFeatureTransformer(
    variables=['order_date', 'delivery_date'],
    features=['year', 'month', 'day', 'dayofweek', 'quarter'],
    drop_original=True
)

# 파이프라인에 추가
pipe = Pipeline([
    ('date_features', date_transformer),
    # ... 다른 전처리 단계들 ...
])

# 또는 단독으로 사용
X_transformed = date_transformer.fit_transform(X)
"""