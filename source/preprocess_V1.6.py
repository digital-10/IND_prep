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

def position_Y_col(cols):
    if Y_COL in cols:  # Y_COL이 있을 때만 remove 실행
        cols_copy = cols.copy()
        cols_copy.remove(Y_COL)
        return cols_copy + [Y_COL]
    else:  # Y_COL이 없으면 변경없이 리턴
        return cols