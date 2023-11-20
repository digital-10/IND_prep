import sys
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

import lightgbm as lgb

from feature_engine import imputation as mdi
from feature_engine import encoding as ce
from feature_engine import outliers as ol


def get_path():
    cur_path = os.getcwd()
    parent_path = os.path.dirname(cur_path)
    return cur_path, parent_path

def model_selection(option):
    if option == 'light':
        return lgb.LGBMClassifier(random_state=0)
    else:
        return GradientBoostingClassifier(random_state=0)


def read_config(config_name):
    import yaml
    try:
        _, parent_path = get_path()
        with open(f'{parent_path}/yml/{config_name}', "r", encoding="utf-8") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            train = cfg['train']
            test = cfg['test']
            Y_col = cfg['Y_col']
            option = cfg['option']
            return train, test, Y_col, option
    except Exception as e:
        print(e)
        return '', '', '', ''


if __name__ == '__main__':
    config_file_name = sys.argv[1]
    print('config_file_name', config_file_name)
    _, parent_path = get_path()

    train, test, Y_col, option = read_config(config_file_name)
    train_df = pd.read_csv(f'{parent_path}/result_data/{train}')
    test_df = pd.read_csv(f'{parent_path}/result_data/{train}')
    model = model_selection(option=option)
    model.fit(train_df.drop(Y_col, axis=1), train_df[Y_col])
    y_pred = model.predict(test_df.drop(Y_col, axis=1))
    print('f1 score', test, f1_score(test_df[Y_col], y_pred, average='macro'))
