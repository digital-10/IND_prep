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