# import sys
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import f1_score
#
# import lightgbm as lgb
#
# from feature_engine import imputation as mdi
# from feature_engine import discretisation as dsc
# from feature_engine import encoding as ce
# from feature_engine import outliers as ol
#
# def write_processed(train, test, naming, drop):
#     df = train.copy()
#     del train
#     df[Y_col] = test.to_list()
#     del test
#     df.to_csv(f'{naming}.csv', index=drop)
#
# def model_selection(option):
#     if option == 'light':
#         return lgb.LGBMClassifier(random_state=0)
#     else:
#         return GradientBoostingClassifier(random_state=0)
#
# def read_config(config_name):
#     import yaml
#     file = ''
#
#     try:
#         with open(config_name, "r", encoding="utf-8") as ymlfile:
#             cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
#             file = cfg['file']
#             null_threshhold = cfg['null_threshhold']
#             FOLD = cfg['FOLD']
#             TEST_SIZE = cfg['TEST_SIZE']
#             Y_col = cfg['Y_col']
#             mixed = cfg['mixed']
#             date_type = cfg['date_type']
#             date_remove = cfg['date_remove']
#             col_remove = cfg['col_remove']
#             y_col_is_null = cfg['y_col_is_null']
#             uids = cfg['uids']
#             drop = cfg['drop']
#             option = cfg['option']
#             return file, null_threshhold, FOLD, TEST_SIZE, Y_col, mixed, date_type, date_remove, col_remove, y_col_is_null, uids, drop, option
#     except Exception as e:
#         print(e)
#         return file
#
#
# if __name__ == '__main__':
#     config_file_name = sys.argv[1]
#     print('config_file_name', config_file_name)
#
#     file, null_threshhold, FOLD, TEST_SIZE, Y_col, mixed, date_type, date_remove, col_remove, y_col_is_null, uids, drop, option = read_config(config_file_name)
#     if len(date_type) == 0:
#         df = pd.read_csv(file)
#     else:
#         df = pd.read_csv(file, parse_dates=date_type)
#     if len(uids) == 0:
#         pass
#     else:
#         df = df.set_index(uids, drop=False)
#     if len(col_remove) == 0:
#         pass
#     else:
#         df = df.drop(col_remove, axis=1)
#
#     labeler = LabelEncoder()
#     df[Y_col] = labeler.fit_transform(df[Y_col])
#
#     cols = df.columns.to_list()
#     null_threshhold_cols = []
#     no_null_cols = []
#
#     for col in cols:
#         null_mean = df[col].isnull().mean()
#         if null_mean >= null_threshhold:
#             null_threshhold_cols.append(col)
#         if null_mean == 0:
#             no_null_cols.append(col)
#
#     cols_stayed = [item for item in cols if item not in null_threshhold_cols]
#     data = df[cols_stayed].copy()
#
#     # numerical: discrete vs continuous
#     date_time = date_type.copy()
#     discrete = [var for var in cols_stayed if
#                 data[var].dtype != 'O' and var != Y_col and var not in date_time and data[var].nunique() < 10]
#     continuous = [var for var in cols_stayed if
#                   data[var].dtype != 'O' and var != Y_col and var not in date_time and var not in discrete]
#
#     # categorical
#     if len(mixed) == 0:
#         categorical = [var for var in cols_stayed if data[var].dtype == 'O' and var != Y_col]
#     else:
#         categorical = [var for var in cols_stayed if data[var].dtype == 'O' and var != Y_col and var not in mixed]
#
#     print('There are {} date_time variables'.format(len(date_time)))
#     print('There are {} discrete variables'.format(len(discrete)))
#     print('There are {} continuous variables'.format(len(continuous)))
#     print('There are {} categorical variables'.format(len(categorical)))
#
#     if y_col_is_null:
#         data_notnull = data[data[Y_col] != data[Y_col].max()].copy()
#     else:
#         data_notnull = data.copy()
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         data_notnull.drop(Y_col, axis=1),
#         data_notnull[Y_col],
#         test_size=TEST_SIZE,
#         random_state=0)
#
#     numberImputer = [item for item in continuous + discrete if item not in no_null_cols]
#     categoricalImputer = [item for item in categorical if item not in no_null_cols]
#
#     pipe = Pipeline([
#         ("median_imputer",
#          mdi.MeanMedianImputer(
#              imputation_method="median", variables=numberImputer),),
#
#         ('imputer_cat',
#          mdi.CategoricalImputer(variables=categorical)),
#
#         ('categorical_encoder',
#          ce.OrdinalEncoder(encoding_method='ordered',
#                            variables=categorical))
#     ])
#
#     pipe.fit(X_train, y_train)
#     X_train_t = pipe.transform(X_train)
#     X_test_t = pipe.transform(X_test)
#
#     model = model_selection(option=option)
#     if date_remove == '':
#         model.fit(X_train_t, y_train)
#     else:
#         model.fit(X_train_t.drop(date_remove, axis=1), y_train)
#
#     if date_remove == '':
#         y_pred = model.predict(X_test_t)
#     else:
#         y_pred = model.predict(X_test_t.drop(date_remove, axis=1))
#     print('f1 score with no scaling', f1_score(y_test, y_pred, average='macro'))
#
#     write_processed(X_test_t, y_test, 'X_test_transform', drop)
#     write_processed(X_train_t, y_train, 'X_train_transform', drop)
#
#     scaler = MinMaxScaler()  # .set_output(transform="pandas")
#
#     if date_remove == '':
#         scaler.fit(X_train_t)
#         X_train_scaled = scaler.transform(X_train_t)
#         X_test_scaled = scaler.transform(X_test_t)
#         write_processed(pd.DataFrame(data=X_train_scaled, columns=X_train_t.columns), y_train, 'X_train_scaled')
#         write_processed(pd.DataFrame(data=X_test_scaled, columns=X_test_t.columns), y_test, 'X_test_scaled')
#     else:
#         scaler.fit(X_train_t.drop(date_remove, axis=1))
#         X_train_scaled = scaler.transform(X_train_t.drop(date_remove, axis=1))
#         X_test_scaled = scaler.transform(X_test_t.drop(date_remove, axis=1))
#         write_processed(
#             pd.DataFrame(data=X_train_scaled, columns=X_train_t.drop(date_remove, axis=1).columns).set_index(
#                 X_train_t.index), y_train, 'X_train_scaled', drop)
#         write_processed(pd.DataFrame(data=X_test_scaled, columns=X_test_t.drop(date_remove, axis=1).columns).set_index(
#             X_test_t.index), y_test, 'X_test_scaled', drop)
#
#         model.fit(X_train_scaled, y_train)
#         y_pred = model.predict(X_test_scaled)
#         print('f1 score with scaling', f1_score(y_test, y_pred, average='macro'))
#
#     trimmer = ol.OutlierTrimmer(
#         variables=continuous,
#         capping_method="iqr",
#         tail="both",
#         fold=FOLD,
#     )
#
#     if date_remove == '':
#         trimmer.fit(X_train_t)
#         X_train_enc = trimmer.transform(X_train_t)
#         y_train_enc = y_train[X_train_enc.index.to_list()]
#         X_test_enc = trimmer.transform(X_test_t)
#         y_test_enc = y_test[X_test_enc.index.to_list()]
#     else:
#         trimmer.fit(X_train_t.drop(date_remove, axis=1))
#         X_train_enc = trimmer.transform(X_train_t.drop(date_remove, axis=1))
#         y_train_enc = y_train[X_train_enc.index.to_list()]
#         X_test_enc = trimmer.transform(X_test_t.drop(date_remove, axis=1))
#         y_test_enc = y_test[X_test_enc.index.to_list()]
#
#     model.fit(X_train_enc, y_train_enc)
#     y_pred = model.predict(X_test_enc)
#     print('f1 score with outlier removed', f1_score(y_test_enc, y_pred, average='macro'))
#
#     write_processed(X_train_enc, y_train_enc, 'X_train_outliered', drop)
#     write_processed(X_test_enc, y_test_enc, 'X_test_outliered', drop)
