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
    )
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