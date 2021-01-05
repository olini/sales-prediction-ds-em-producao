import pickle
import pandas as pd
import requests
from flask import Flask, request, Response
from ml_utils import Ml_utils
import bot_utils

# loading model
xgb_model = pickle.load(open('model/xgb_reg_model.pkl', 'rb'))

# initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    data_json = request.get_json()
    
    if data_json: # there is data
        if isinstance(data_json, dict): # one row
            df_raw = pd.DataFrame(data_json, index=[0])
        else: # multiple rows
            df_raw = pd.DataFrame(data_json, columns=data_json[0].keys())
            
        # instantiate ml_utils class
        ml_utils = Ml_utils()
        
        # filter data
        df1 = ml_utils.filter_data(df_raw)
        
        # clean data
        df2 = ml_utils.clean_data(df1)
        
        # feature engineering
        df3 = ml_utils.feature_engineering(df2)
        
        # prepare data for ml model
        df4 = ml_utils.prepare_data(df3)
        
        # get predictions
        df_response = ml_utils.get_prediction(xgb_model, df1, df4)
        
        df_response = df_response.to_json(orient='records', date_format='iso')
        
        return df_response
    else:
        return Response('{}', status=200, mimetype='application/json')

@app.route('/predict-bot', methods=['POST'])
def predict_bot():
    message = request.get_json()
    chat_id, store_id = bot_utils.parse_message(message)

    if store_id != 'error':
        # load test data
        df_test = bot_utils.load_dataset(store_id)
        
        if not df_test.empty:
            df_response = predict(df_test)
            df_response_agg = df_response.groupby('store')['prediction'].sum().reset_index()
            
            msg = f"Store Number {df_response_agg['store'].values[0]} will sell R${df_response_agg['prediction'].values[0]:,.2f}"
        else:
            msg = 'Store Not Available'
    else:
        msg = 'Wrong Store Id. Pass it with a / followed by the Store Id. Example: /22 for Store 22'
    
    bot_utils.send_message(chat_id, msg)
    return Response('Ok', status=200)

def predict(df_test):        
    # instantiate ml_utils class
    ml_utils = Ml_utils()

    # filter data
    df1 = ml_utils.filter_data(df_test)

    # clean data
    df2 = ml_utils.clean_data(df1)

    # feature engineering
    df3 = ml_utils.feature_engineering(df2)

    # prepare data for ml model
    df4 = ml_utils.prepare_data(df3)

    # get predictions
    df_response = ml_utils.get_prediction(xgb_model, df1, df4)

    return df_response

if __name__ == '__main__':
    app.run('0.0.0.0')