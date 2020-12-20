# import necessary libraries
import json
import requests
import pandas as pd

def load_datasets(store_ids):
    # load test dataset
    df_test_raw = pd.read_csv('../data/test.csv')

    # load store dataset
    df_store_raw = pd.read_csv('../data/store.csv')

    # merge with store dataset
    df_test = pd.merge(df_test_raw, df_store_raw, how='left', on='Store')

# choose store for prediction
df_test = df_test[df_test['Store'].isin([20, 22, 23])]

# remove closed days
df_test = df_test[df_test['Open'] != 0]
df_test = df_test[~df_test['Open'].isnull()]

# drop unecessary column
df_test = df_test.drop('Id', axis=1)

# convert DataFrame to json
data_json = json.dumps(df_test.to_dict(orient='records'))

# API Call
url = 'https://olini-rossmann-sales-pred.herokuapp.com/predict'
header = {'Content-type': 'application/json'}

r = requests.post(url, data=data_json, headers=header)
print(f'Status Code {r.status_code}')

df_response = pd.DataFrame(r.json(), columns=r.json()[0].keys())

df_response_group = df_response.groupby('store')['prediction'].sum().reset_index()

for i in range(len(df_response_group)):
    print(f"Store Number {df_response_group.loc[i, 'store']} will sell R${df_response_group.loc[i, 'prediction']:,.2f}")