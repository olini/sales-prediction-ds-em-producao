import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Ml_utils(object):
    def __init__(self):
        self.store_type_encoder = pickle.load(open('data_transformation/encoder_store_type.pkl', 'rb'))
        self.competition_distance_scaler = pickle.load(open('data_transformation/scaler_competition_distance.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open('data_transformation/scaler_competition_time_month.pkl', 'rb'))
        self.promo_time_week_scaler = pickle.load(open('data_transformation/scaler_promo_time_week.pkl', 'rb'))
    
    
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
        df['competition_open_since_month'].fillna(df['date'].dt.month, inplace=True)
        df['competition_open_since_month'] = df['competition_open_since_month'].astype(int)
        # competition_open_since_year
        df['competition_open_since_year'].fillna(df['date'].dt.year, inplace=True)
        df['competition_open_since_year'] = df['competition_open_since_year'].astype(int)
        # promo2_since_week
        df['promo2_since_week'].fillna(df['date'].dt.isocalendar().week, inplace=True)
        df['promo2_since_week'] = df['promo2_since_week'].astype(int)
        # promo2_since_year
        df['promo2_since_year'].fillna(df['date'].dt.year, inplace=True)
        df['promo2_since_year'] = df['promo2_since_year'].astype(int)
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
    
    
    def feature_engineering(self, df):
        # year
        df['year'] = df['date'].dt.year

        # month
        df['month'] = df['date'].dt.month

        # day
        df['day'] = df['date'].dt.day

        # week of year
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # year-week
        df['year_week'] = df['date'].dt.strftime('%Y-%W')

        # competition_since
        df['competition_since'] = df.apply(lambda x: 
                                           datetime.datetime(
                                               year=x['competition_open_since_year'],
                                               month=x['competition_open_since_month'], 
                                               day=1), axis=1)
        df['competition_time_month'] = ((df['date'] - df['competition_since'])/30).apply(lambda x: x.days).astype(int)

        # promo since
        df['promo_since'] = df['promo2_since_year'].astype(str) + '-' + df['promo2_since_week'].astype(str)
        df['promo_since'] = df['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df['promo_time_week'] = ((df['date'] - df['promo_since'])/7).apply(lambda x: x.days).astype(int)

        # assortment
        df['assortment'] = df['assortment'].apply(lambda x: 'basic' if x == 'a' 
                                                        else 'extra' if x == 'b' 
                                                        else 'extended' if x == 'c' 
                                                        else 'not_found')

        # state holiday
        df['state_holiday'] = df['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' 
                                                              else 'easter_holiday' if x == 'b' 
                                                              else 'christmas' if x == 'c' 
                                                              else 'regular_day')
        
        return df
    
    
    def filter_data(self, df):
        # filter rows
        df = df[(df['Open'] == 1)]
        
        return df
    
    
    def prepare_data(self, df):
        # competition_distance - Robust Scaler
        df['competition_distance'] = self.competition_distance_scaler.transform(df[['competition_distance']].values)

        # competition_time_month - RObust Scaler
        df['competition_time_month'] = self.competition_time_month_scaler.transform(df[['competition_time_month']].values)

        # promo_time_week - Robust Scaler
        df['promo_time_week'] = self.promo_time_week_scaler.transform(df[['promo_time_week']].values)

        # store_type - Label Enconding
        df['store_type'] = self.store_type_encoder.transform(df['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df['assortment'] = df['assortment'].map(assortment_dict)
        
        # day of week
        df['day_of_week_sin'] = df['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi/7)))
        df['day_of_week_cos'] = df['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi/7)))

        # month
        df['month_sin'] = df['month'].apply(lambda x: np.sin(x * (2. * np.pi/12)))
        df['month_cos'] = df['month'].apply(lambda x: np.cos(x * (2. * np.pi/12)))

        # day
        df['day_sin'] = df['day'].apply(lambda x: np.sin(x * (2. * np.pi/30)))
        df['day_cos'] = df['day'].apply(lambda x: np.cos(x * (2. * np.pi/30)))

        # week of year
        df['week_of_year_sin'] = df['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi/52)))
        df['week_of_year_cos'] = df['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi/52)))
        
        # selected columns
        features_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
                             'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month',
                             'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                             'week_of_year_sin', 'week_of_year_cos']
        
        return df[features_selected]
    
    
    def get_prediction(self, model, original_data, test_data):
        # prediction
        pred = model.predict(test_data)
        
        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data