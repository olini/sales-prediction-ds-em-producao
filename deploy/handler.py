import pickle
import pandas as pd
from flask import Flask, request, Response
from ml_utils import Ml_utils

# loading model
xgb_model = pickle.load(open('model/xgb_reg_model.pkl', 'rb'))

# initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
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
        
        return df_response
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)