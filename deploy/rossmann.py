import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann(object):
    def __init__(self):
        store_type_encoder = pickle.load(open('data_transformation/encoder_store_type.pkl'))
        competition_distance_scaler = pickle.load(open('data_transformation/scaler_competition_distance.pkl'))
        competition_time_month_scaler = pickle.load(open('data_transformation/scaler_competition_time_month.pkl'))
        promo_time_week_scaler = pickle.load(open('data_transformation/scaler_promo_time_week.pkl'))
    
    
    def clean_data(self, df):
        # rename columns
        old_cols_name = list(df.columns) 
        snake_case_func = lambda x: inflection.underscore(x)
        new_cols_name = list(map(snake_case_func, old_cols_name))
        df.columns = new_cols_name
        
        # change data type for the field date, from object to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # fill NAs
        # competition_distance
        df['competition_distance'].fillna(200000, inplace=True)
        # competition_open_since_month
        df['competition_open_since_month'].fillna(df['date'].dt.month, inplace=True).astype(int)
        # competition_open_since_year
        df['competition_open_since_year'].fillna(df['date'].dt.year, inplace=True).astype(int)
        # promo2_since_week
        df['promo2_since_week'].fillna(df['date'].dt.isocalendar().week, inplace=True).astype(int)
        # promo2_since_year
        df['promo2_since_year'].fillna(df['date'].dt.year, inplace=True).astype(int)
        # promo_interval
        month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df['promo_interval'].fillna(0, inplace=True)
        df['date_month'] = df['date'].dt.month.map(month_dict)
        # TO DO - creation of is_promo feature inside data cleaning
        df['is_promo'] = df[['promo_interval', 'date_month']].apply(lambda x:
                                                                          0 if x['promo_interval'] == 0 else 
                                                                          1 if x['date_month'] in x['promo_interval'].split(',') 
                                                                          else 0, axis=1)
        
        return df
        