import pickle
import pandas as pd
from flask import Flask, request, Response

# loading model
xgb_model = pickle.load(open('model/xgb_reg_model.pkl', 'rb'))

# initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def main():
    


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)