import requests
import pandas as pd

TOKEN = '1598094502:AAEiwve1iGjOt_rFSCVaNXcC8XKuFgH1yUs'

def parse_message(message):
    chat_id = message['message']['chat']['id']
    text = message['message']['text']
    
    store_id = text.replace('/', '')
    
    try:
        store_id = int(store_id)
    except ValueError:
        store_id = 'error'
        
    return chat_id, store_id

def load_dataset(store_id):
    # load datasets
    df_test_raw = pd.read_csv('data/test.csv')
    df_store_raw = pd.read_csv('data/store.csv')
    
    # merge datasets
    df_test = pd.merge(df_test_raw, df_store_raw, how='left', on='Store')
    
    # get only chosen store for prediction
    df_test = df_test[df_test['Store'] == store_id]
    
    return df_test

def send_message(chat_id, msg):
    url = f'https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}'
    r = requests.post(url, json={'text': msg})
    print(f'Status Code {r.status_code}')
    
    return
        
    