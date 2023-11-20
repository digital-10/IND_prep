import sys
import pandas as pd
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def get_path():
    cur_path = os.getcwd()
    parent_path = os.path.dirname(cur_path)
    return cur_path, parent_path


def model_selection(option='logic'):
    if option == 'light':
        return lgb.LGBMClassifier(random_state=0)
    else:
        return LogisticRegression(random_state=0)


def split_train_test(df, configs):
    df = df.copy()
    X = df.drop(columns=configs['y_col'][0])
    y = df[configs['y_col'][0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=configs['y_col'][0])
    return X_train, X_test, y_train, y_test


def metrics(y_test, pred, option):
    y_test = y_test.copy()
    pred = pred.copy()
    accuracy = round(accuracy_score(y_test, pred), 2)
    precision = round(precision_score(y_test, pred), 2)
    recall = round(recall_score(y_test, pred), 2)
    f1 = round(f1_score(y_test, pred), 2)
    print(option, "f1 점수:", f1, "정확도:", accuracy, "정밀도:", precision, "재현율:", recall)
    print(confusion_matrix(y_test, pred))


def do_train(X_train, X_test, y_train, y_test, option):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    model = model_selection(option)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics(y_test, y_pred, option)


if __name__ == '__main__':
    # arv 예1: credit argumet_credit.xlsx
    # arv 예2: metro argumet_metro.xlsx

    error_messages = []
    folder_name = sys.argv[1]  #ex: credit argumet_credit.xlsx
    config_file_name = sys.argv[2]
    cur_path = os.getcwd()
    parent = os.path.abspath(os.path.join(cur_path, os.pardir))
    config_file = os.path.join(parent, os.path.join('config', f'{config_file_name}'))
    configs = pd.read_excel(config_file, header=None).set_index(0).T
    configs = configs.to_dict('list')
    ori_file_name = configs['file_name'][0]
    configs['file_name'][0] = os.path.join(parent, os.path.join('data_preprocessed', configs['file_name'][0]))

    dataset_folder = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}'))
    model = model_selection('logic')

    files = os.listdir(dataset_folder)
    for afile in files:
        if afile.startswith('draft'):
            try:
                df_draft = pd.read_csv(os.path.join(dataset_folder, f'{afile}'))
                print('실행할 파일명:', afile)
                df = pd.read_csv(os.path.join(dataset_folder, f'{afile}'))
                X_train, X_test, y_train, y_test = split_train_test(df, configs)
                do_train(X_train, X_test, y_train, y_test, 'logic')
            except Exception as e:
                error_messages.append([afile, e])
                print(error_messages)
        elif afile.startswith('Scaled_Xtrain'):
            scaled_x_train_file = afile
            X_train_scaled = pd.read_csv(os.path.join(dataset_folder, f'{afile}'))
        elif afile.startswith('Scaled_Xtest'):
            scaled_x_test_file = afile
            X_test_scaled = pd.read_csv(os.path.join(dataset_folder, f'{afile}'))
        elif afile.startswith('Xtrain'):
            x_train_file = afile
            X_train = pd.read_csv(os.path.join(dataset_folder, f'{afile}'))
        elif afile.startswith('Xtest'):
            x_test_file = afile
            X_test = pd.read_csv(os.path.join(dataset_folder, f'{afile}'))
        elif afile.startswith('ytrain'):
            y_train_file = afile
            y_train = pd.read_csv(os.path.join(dataset_folder, f'{afile}'))
        elif afile.startswith('ytest'):
            y_test_file = afile
            y_test = pd.read_csv(os.path.join(dataset_folder, f'{afile}'))
    print()
    print()

    print('1차 전처리 실행 파일:', x_train_file, x_test_file, y_train_file, y_test_file)
    print()
    do_train(X_train, X_test, y_train, y_test, 'logic')
    print()
    do_train(X_train, X_test, y_train, y_test, 'light')

    print()
    print()

    print('스케일링 전처리 실행 파일:', scaled_x_train_file, scaled_x_test_file, y_train_file, y_test_file)
    print()
    do_train(X_train_scaled, X_test_scaled, y_train, y_test, 'logic')
    print()
    do_train(X_train_scaled, X_test_scaled, y_train, y_test, 'light')

    print('successful Ending')