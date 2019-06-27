# Submission script for AIVIVN's 5th competition: Server bandwidth and max user prediction
# Created by: Le Anh Tho - anhtholee 
# Model used: XGBoost
# Features used:
# - Time components
# - Special day impact & holiday binary feature
# - Zone-specific medians & autocorr
# - Predictions using linear regression (used as a feature for XGBoost)
# Import libraries
import numpy as np
import pandas as pd 
from functools import reduce
import xgboost as xgb # XGBoost model
from sklearn.linear_model import Ridge # Linear regression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from utils.preprocessing import *
import sys,os

BASE_DIR = os.path.join('data')
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'test_id.csv')
if __name__ == "__main__":
    # Load train and test data
    df, test_df = load_csv(TRAIN_PATH), load_csv(TEST_PATH)
    
    # =========== FEATURE ENGINEERING ===========
    # Time features
    df, test_df = add_time_features(df), add_time_features(test_df)
    # Special events featuers
    df, test_df = add_special_days_features(df), add_special_days_features(test_df)
    # Zone features
    zfeatures = ['median_user_1m', 'median_bw_1m', 'median_user_3m', 'median_bw_3m', 'median_user_6m', 'median_bw_6m', 'median_user_1y', 'median_bw_1y']
    aufeatures = ['lag_user_1d', 'lag_user_3d', 'lag_user_1w', 'lag_bw_1d', 'lag_bw_3d', 'lag_bw_1w']   
    zones, zones_autocorr = zone_features(df, zfeatures, aufeatures)

    features = ['zone_code', 'hour_id', 'dow_norm', 'month', 'doy', 'year', 'day', 'week', 'abnormal_bw', 'abnormal_u', 'holiday']
    rfeatures = ['ridge_bw', 'ridge_u']

    # Merge the data with the zone features 
    dfr = pd.merge(df,zones,on='zone_code')
    dfr = dfr.merge(zones_autocorr, how='inner', on=['zone_code']).sort_values(by=['update_time','zone_code'], ascending=[True,True])

    test = pd.merge(test_df,zones,on='zone_code')
    test = test.merge(zones_autocorr, how='inner', on=['zone_code'])

    # Label encoding for zone codes
    le = LabelEncoder()
    le.fit(dfr['zone_code'])
    dfr['zone_code'] = le.transform(dfr['zone_code'])
    test['zone_code'] = le.transform(test['zone_code'])

    # Add one more feature: linear regression prediction
    clf1 = Ridge(alpha=1)
    clf1.fit(dfr[features + zfeatures], np.log1p(dfr['bandwidth_total']))
    dfr['ridge_bw'] = clf1.predict(dfr[features + zfeatures])
    test['ridge_bw'] = clf1.predict(test[features + zfeatures])

    clf2 = Ridge(alpha=1)
    clf2.fit(dfr[features + zfeatures], np.log1p(dfr['max_user']))
    dfr['ridge_u'] = clf2.predict(dfr[features + zfeatures])
    test['ridge_u'] = clf2.predict(test[features + zfeatures])

    # =========== MODELLING ===========
    # Init the models
    m1 = xgb.XGBRegressor(
        n_jobs = -1,
        n_estimators = 500,
        eta = 0.01,
        max_depth = 5,
        min_child_weight = 1,
        booster = 'gbtree',
        subsample = 0.8,
        colsample_bytree = 0.7,
        tree_method = 'exact',
        silent = 0,
        gamma = 0,
        random_state = 1023
      )

    m2 = xgb.XGBRegressor(
        n_jobs = -1,
        n_estimators = 500,
        eta = 0.01,
        max_depth = 7,
        min_child_weight = 1,
        booster = 'gbtree',
        subsample = 0.8,
        colsample_bytree = 0.7,
        tree_method = 'exact',
        silent = 0,
        gamma = 0,
        random_state = 1023
    )

    # Fit the model to the data
    for i, col in enumerate(['bandwidth_total', 'max_user']):
        X_train = dfr[features+zfeatures+rfeatures+aufeatures]
        y_train = dfr[col]
        if i == 0:
            m = m1
            m.fit(X_train, np.log1p(y_train), eval_metric='mae')
            test[col] = np.expm1(m.predict(test[features+zfeatures+rfeatures+aufeatures]))
        else:
            m = m2
            m.fit(X_train, np.log1p(y_train), eval_metric='mae')
            test[col] = np.expm1(m.predict(test[features+zfeatures+rfeatures+aufeatures]))
            
    # =========== SUBMISSION ===========
    test['bandwidth_total'] = test['bandwidth_total'].round(2)
    test['max_user'] = test['max_user'].round()
    test['label'] = test['bandwidth_total'].astype(str) + ' ' + test['max_user'].astype(int).astype(str)
    test[['id', 'label']].to_csv('submission.csv', index=False)
