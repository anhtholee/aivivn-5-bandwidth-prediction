# Some utilites functions for loading the data, adding features
import numpy as np 
import pandas as pd 
from functools import reduce
from sklearn.preprocessing import MinMaxScaler

def load_csv(path):
    """Load dataframe from a csv file
    
    Args:
        path (STR): File path
    """
    # Load the file
    df = pd.read_csv(path)
    # Lowercase column names
    df.rename(columns=lambda x: x.lower().strip(), inplace=True)

    return df

def add_time_features(df):
    """Add time features for the data
    
    Args:
        df (DataFrame): Input dataframe
    Return: the modified df
    """
    df['ds'] = pd.to_datetime(df['update_time']) + df['hour_id'].astype('timedelta64[h]')
    df['dow'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['doy'] = df['ds'].dt.dayofyear
    df['year'] = df['ds'].dt.year
    df['day'] = df['ds'].dt.day
    df['week'] = df['ds'].dt.week

    # Normalise day of week col
    week_period = 7 / (2 * np.pi)
    df['dow_norm'] = df.dow.values / week_period

    return df 

def add_special_days_features(df):
    """Add special events and holidays features
    
    Args:
        df (DataFrame): Input dataframe
    Return: the modified df
    """
    # Days when there were sudden decrease/increase in bandwidth/max users
    range1 = pd.date_range('2018-02-10', '2018-02-27')
    range2 = pd.date_range('2019-01-30', '2019-02-12')
    abnormals = range1.union(range2)

    # Init 2 new columns
    df['abnormal_bw'], df['abnormal_u'] = 0,0
    # Set the abnormal weights for each zone (negative if decrease, positive if increase)
    # For total bandwidth
    df.loc[df['zone_code'].isin(['ZONE01']) ,'abnormal_bw'] = df[df['zone_code'].isin(['ZONE01'])].update_time.apply(lambda date: -1 if pd.to_datetime(date) in abnormals else 0)
    df.loc[df['zone_code'].isin(['ZONE02']) ,'abnormal_bw'] = df[df['zone_code'].isin(['ZONE02'])].update_time.apply(lambda date: 1 if pd.to_datetime(date) in abnormals else 0)
    df.loc[df['zone_code'].isin(['ZONE03']) ,'abnormal_bw'] = df[df['zone_code'].isin(['ZONE03'])].update_time.apply(lambda date: 0.2 if pd.to_datetime(date) in abnormals else 0)

    # For max users
    df.loc[df['zone_code'].isin(['ZONE01']) ,'abnormal_u'] = df[df['zone_code'].isin(['ZONE01'])].update_time.apply(lambda date: -1 if pd.to_datetime(date) in abnormals else 0)
    df.loc[df['zone_code'].isin(['ZONE02']) ,'abnormal_u'] = df[df['zone_code'].isin(['ZONE02'])].update_time.apply(lambda date: 1 if pd.to_datetime(date) in abnormals else 0)
    df.loc[df['zone_code'].isin(['ZONE03']) ,'abnormal_u'] = df[df['zone_code'].isin(['ZONE03'])].update_time.apply(lambda date: 0.8 if pd.to_datetime(date) in abnormals else 0)

    # Holidays
    holidays = pd.to_datetime(['2018-01-01', 
                           '2018-02-14', '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18', '2018-02-19', '2018-02-20',
                           '2018-03-27', '2018-04-30', '2018-05-01', '2018-09-02', '2018-09-03', '2018-12-31',
                           '2019-01-01', '2019-02-04', '2019-02-05', '2019-02-06', '2019-02-07', '2019-02-08',
                           '2019-04-15', 
                           '2019-04-29', '2019-04-30', '2019-05-01', '2019-09-02',
                          ])
    df['holiday'] = df.update_time.apply(lambda date: 1 if pd.to_datetime(date) in holidays else 0)

    return df

def zone_features(df, zfeatures, aufeatures):
    """Create zone features from the data
    
    Args:
        df (DataFrame): Input dataframe
        zfeatures (list): List of zone median features
        aufeatures (list): List of zone autocorr features
    Return: 2 dataframes
    """

    # Medians from the last 1,3,6,12 months
    zones_1y = df[(df['ds'] >= '2018-03-09') & (df['ds'] < '2019-03-10')].groupby(['zone_code'], as_index=False).agg({
        'max_user': 'median', 
        'bandwidth_total': 'median' 
    })
    zones_1y.columns = ['zone_code','median_user_1y','median_bw_1y']

    zones_1m = df[(df['ds'] >= '2019-02-09') & (df['ds'] < '2019-03-10')].groupby(['zone_code'], as_index=False).agg({
        'max_user': 'median', 
        'bandwidth_total': 'median' 
    })
    zones_1m.columns = ['zone_code','median_user_1m','median_bw_1m']

    zones_3m = df[(df['ds'] >= '2018-12-09') & (df['ds'] < '2019-03-10')].groupby(['zone_code'], as_index=False).agg({
        'max_user': 'median', 
        'bandwidth_total': 'median' 
    })
    zones_3m.columns = ['zone_code','median_user_3m','median_bw_3m']

    zones_6m = df[(df['ds'] >= '2018-09-09') & (df['ds'] < '2019-03-10')].groupby(['zone_code'], as_index=False).agg({
        'max_user': 'median',
        'bandwidth_total': 'median' 
    })
    zones_6m.columns = ['zone_code','median_user_6m','median_bw_6m']

    # Autocorrelation features
    zones_autocorr = df[(df['ds'] >= '2018-12-09') & (df['ds'] < '2019-03-10')].groupby(['zone_code'], as_index=False).agg({
        'max_user': {
            'lag_user_1d' :lambda x: pd.Series.autocorr(x, 24),
            'lag_user_3d' :lambda x: pd.Series.autocorr(x, 3*24),
            'lag_user_1w' :lambda x: pd.Series.autocorr(x, 24*7),
        }, 
        'bandwidth_total': {
            'lag_bw_1d' :lambda x: pd.Series.autocorr(x, 24),
            'lag_bw_3d' :lambda x: pd.Series.autocorr(x, 3*24),
            'lag_bw_1w' :lambda x: pd.Series.autocorr(x, 24*7),
        }
    }).fillna(0)
    zones_autocorr.columns.droplevel()
    zones_autocorr.reset_index()
    zones_autocorr.columns = ['zone_code','lag_user_1d','lag_user_3d','lag_user_1w','lag_bw_1d','lag_bw_3d','lag_bw_1w']
    zones = reduce(lambda x,y: pd.merge(x,y, on='zone_code', how='inner'), [zones_1m, zones_3m, zones_6m, zones_1y])

    # Scale the zone features
    scale1, scale2 = MinMaxScaler(), MinMaxScaler()
    zones[zfeatures] = scale1.fit_transform(zones[zfeatures])
    zones_autocorr[aufeatures] = scale2.fit_transform(zones_autocorr[aufeatures])

    return zones, zones_autocorr