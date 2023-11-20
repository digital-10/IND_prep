from elasticsearch import Elasticsearch
from IPython.display import display
import pandas as pd
import time
import datetime
from datetime import timedelta
import os

def utc_time():  # @timestamp timezone을 utc로 설정하여 kibana로 index 생성시 참조
    return datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S') + 'Z'

def p2w(df):
    df = df.copy()  
    try:
        docu = df.to_dict()
        print(docu)
        res = es.index(index=index, body=docu)
    except Exception as e:
            print(e)
            
ELASTIC_PASSWORD = "elastic"
es = Elasticsearch(
    hosts="http://192.168.10.17:9200",    
    basic_auth=("elastic", ELASTIC_PASSWORD) )
# )
es.info()

def delete_es(index):
    es.indices.delete(index=index, ignore=[400, 404])
    
index = "LC_DCS_HEAT_DEMAND_STAT_HOUR".lower()

delete_es(index)

if es.indices.exists(index=index):
	pass
else:
	es.indices.create(index=index) #, body=mapping)
    
data_path = r"C:\Users\Digitalship_PC\pydev\digitalship\dcat\input_data\iot"
files = os.listdir(data_path)
files.sort()
df = pd.DataFrame()
for f in files[0:30]:
    temp = pd.read_csv(rf'{data_path}\{f}')
    temp['DATE'] = pd.to_datetime(temp['DATE'], format='%Y-%m-%d %H:%M:%S') # - timedelta(hours=9)
    temp['DATE'] = temp['DATE'].dt.strftime('%Y-%m-%dT%H:%M:%S.000')# + 'Z'    
    temp.columns = temp.columns.str.lower()
    col = ['date', 'heat_demand_hour_mean', 'heat_demand_hour_max', 'heat_demand_hour_min', 'heat_demand_hour_total', 'reg_dt']
    temp = temp[col]
    df = pd.concat([df, temp])
    
df.to_csv(r"C:\Users\Digitalship_PC\pydev\digitalship\source\elastic_data.csv", index=False)
display(df)

data_path = r"C:\Users\Digitalship_PC\pydev\digitalship\dcat\input_data\iot"
files = os.listdir(data_path)
files.sort()
for f in files[:30]:
    temp = pd.read_csv(rf'{data_path}\{f}')
    temp['DATE'] = pd.to_datetime(temp['DATE'], format='%Y-%m-%d %H:%M:%S') - timedelta(hours=9)
    temp['DATE'] = temp['DATE'].dt.strftime('%Y-%m-%dT%H:%M:%S.000')# + 'Z'    
    temp.columns = temp.columns.str.lower()
    col = ['date', 'heat_demand_hour_mean', 'heat_demand_hour_max', 'heat_demand_hour_min', 'heat_demand_hour_total', 'reg_dt']
    temp = temp[col]
    for idx, row in temp.iterrows():    
        display(row)
        p2w(row)
    time.sleep(1)
    
print('Completed')