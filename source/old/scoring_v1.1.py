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


def model_selection(option='logistic'):
    if option == 'light':
        return lgb.LGBMClassifier(random_state=0)
    else:
        return LogisticRegression(random_state=0)


def split_train_test(df, configs):
    df = df.copy()
    X = df.drop(columns=configs['y_col'][0])
    y = df[configs['y_col'][0]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test


def metrics(y_test, pred, algo, file):    
    y_test = y_test.copy()
    pred = pred.copy()
    if len(set(y_test)) <= 2:
        accuracy = round(accuracy_score(y_test, pred), 2)
        precision = round(precision_score(y_test, pred), 2)
        recall = round(recall_score(y_test, pred), 2)
        f1 = round(f1_score(y_test, pred), 2)
    else:
        accuracy = round(accuracy_score(y_test, pred), 2)
        precision = round(precision_score(y_test, pred, average='micro'), 2)
        recall = round(recall_score(y_test, pred, average='micro'), 2)
        f1 = round(f1_score(y_test, pred, average='micro'), 2)
    
    print(algo, file, "f1 점수:", f1, "정확도:", accuracy, "정밀도:", precision, "재현율:", recall)
    print(confusion_matrix(y_test, pred))
    
    return pd.DataFrame(data=[[algo, file, f1, accuracy, precision, recall]], \
                        columns=['알고리즘', '파일명', 'F1 score', 'Accuracy', 'Precision', 'Recall'])


def train_metrics(X_train, X_test, y_train, y_test, algo, file):
    X_train, X_test, y_train, y_test = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    model = model_selection(algo)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = metrics(y_test, y_pred, algo, file)
    return score


if __name__ == '__main__':
    # arv 예1: credit argumet_credit.xlsx
    # arv 예2: metro argumet_metro.xlsx
    
    # 학습데이터셋 마다 다르게 나올 score 를 기록할 df 정의
    scores = pd.DataFrame()
    
    try:
        error_messages = []
        folder_name = sys.argv[1]  #ex: credit argumet_credit.xlsx
        config_file_name = sys.argv[2]
        cur_path = os.getcwd()
        parent = os.path.abspath(os.path.join(cur_path, os.pardir))
        config_file = os.path.join(parent, os.path.join('config', f'{config_file_name}'))
        configs = pd.read_excel(config_file, header=None).set_index(0).T
        configs = configs.to_dict('list')
        ori_file_name = configs['file_name'][0]
        Y_COL = configs['y_col'][0]
        # configs['file_name'][0] = os.path.join(parent, os.path.join('data_preprocessed', configs['file_name'][0]))
    
        dataset_folder_imputed = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}/imputed'))
        dataset_folder_scaled = os.path.join(parent, os.path.join('data_preprocessed', f'{folder_name}/scaled'))

        files = os.listdir(dataset_folder_imputed)        
        files.sort()

        for afile in files:
            print('file_name:', afile)
            if afile.startswith('draft'):
                try:
                    df_draft = pd.read_csv(os.path.join(dataset_folder_imputed, f'{afile}'))
                    print('\n1차 전처리 실행 파일:', afile)
                    df = pd.read_csv(os.path.join(dataset_folder_imputed, f'{afile}'))
                    X_train, X_test, y_train, y_test = split_train_test(df, configs)
                    train_metrics(X_train, X_test, y_train, y_test, 'logic', afile)
                except Exception:
                    message = '학습불가'
                    error_messages.append([afile, message])
                    print(error_messages)
                    continue
            else:
                df = pd.read_csv(os.path.join(dataset_folder_imputed, f'{afile}'))
                X_train = df[df['split']=='train'].drop(columns=[Y_COL, 'split'])
                X_test  = df[df['split']=='test'].drop(columns=[Y_COL, 'split'])
                y_train = df[df['split']=='train'][Y_COL]
                y_test  = df[df['split']=='test'][Y_COL]

            print()
            print()
        
            print('1차 전처리 실행 파일:', afile)
            print()
            algo = 'logistic'
            score = train_metrics(X_train, X_test, y_train, y_test, algo, afile)
            scores = scores.append(score)
            print()

            # algo = 'light'
            # score = train_metrics(X_train, X_test, y_train, y_test, algo, afile)
            # scores = scores.append(score)
        
            # print()
            # print()
            
        files = os.listdir(dataset_folder_scaled)
        files.sort()

        for afile in files:        
            print('스케일링 전처리 실행 파일:', afile)
            df = pd.read_csv(os.path.join(dataset_folder_scaled, f'{afile}'))
            X_train_scaled = df[df['split']=='train']
            y_train = X_train_scaled[Y_COL]
            X_train_scaled =  X_train_scaled.drop(columns=[Y_COL, 'split'])
            X_test_scaled = df[df['split']=='test']
            y_test = X_test_scaled[Y_COL]
            X_test_scaled =  X_test_scaled.drop(columns=[Y_COL, 'split'])
                        
            algo = 'logistic'                                       
            score = train_metrics(X_train, X_test, y_train, y_test, algo, afile)
            scores = scores.append(score)
            print()
            
            # algo = 'light'                                       
            # score = train_metrics(X_train, X_test, y_train, y_test, algo, afile)
            # scores = scores.append(score)
            # print()
        
        scores = scores.sort_values('F1 score', ascending=False).reset_index(drop=True)
        
        # 결과 저장
        scores_folder = os.path.join(parent, os.path.join('scores', f'{folder_name}/scores.csv'))
        scores.to_csv(scores_folder, index=False, encoding='cp949')
        print('Completed.')
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print('비정상종료', e)
        print(exc_type, exc_tb.tb_lineno)
        