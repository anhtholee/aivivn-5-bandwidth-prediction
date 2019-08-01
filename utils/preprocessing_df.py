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

def fill_missing_values(df):
    """Fill the missing data points
    
    Args:
        df: Input dataframe
    Return: the modified dataframe
    """
    # Get datetime col
    df['ds'] = pd.to_datetime(df['update_time']) + df['hour_id'].astype('timedelta64[h]')
    pdlist = []
    for z in df.zone_code.unique():
        zone = df[df['zone_code'] == z]
        r = pd.date_range(zone.ds.min(), zone.ds.max(), freq='H')
        ds_range = pd.DataFrame({'ds': r, 'zone_code': z})
        zone_merged = ds_range.merge(zone, how='left', on=['ds', 'zone_code'])
        zone_merged['hour_id'] = zone_merged['ds'].dt.hour
        
        # Fill the null values 
        for col in ['bandwidth_total', 'max_user']:
            for index, row in zone_merged[zone_merged[col].isnull()].iterrows():
                shifted_index = index - (24*7)
                flag = True
                while flag:
                    fill_val = zone_merged.loc[shifted_index, col]
                    if pd.isnull(fill_val):
                        shifted_index -= (24*7)
                        continue
                    zone_merged.loc[index, col] = fill_val
                    flag = False

        pdlist.append(zone_merged)

    out = pd.concat(pdlist)
    out.drop(['update_time'], axis=1, inplace=True)
    assert not out.isnull().values.any(), 'Error in asserting. There are still nans.'
    return out

def add_time_features(df, test=False):
    """Add time features for the data
    
    Args:
        df (DataFrame): Input dataframe
    Return: the modified df
    """
    if test:
        df['ds'] = pd.to_datetime(df['update_time']) + df['hour_id'].astype('timedelta64[h]')
    else:
        df['update_time'] = df['ds'].dt.date
    df['dow'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['doy'] = df['ds'].dt.dayofyear
    df['year'] = df['ds'].dt.year
    df['day'] = df['ds'].dt.day
    df['week'] = df['ds'].dt.week
    # df['weekend'] = df['dow'] // 5 == 1

    # Normalise day of week col
    week_period = 7 / (2 * np.pi)
    df['dow_norm'] = df.dow.values / week_period

    return df 

def add_time_periods(df):
    """Add time periods of a day 
    
    Args:
        df (DataFrame): Input dataframe
    Return: the modified df
    """
    df['hour_id'] = pd.to_numeric(df['hour_id'])
    conditions = [
        (df['hour_id'] >= 21) | (df['hour_id'] < 1),
        (df['hour_id'] >= 1) & (df['hour_id'] < 6),
        (df['hour_id'] >= 6) & (df['hour_id'] < 11),
        (df['hour_id'] >= 11) & (df['hour_id'] < 14),
        (df['hour_id'] >= 14) & (df['hour_id'] < 17),
        (df['hour_id'] >= 17) & (df['hour_id'] < 19),
        (df['hour_id'] >= 19) & (df['hour_id'] < 21),
    ]
    choices = ['21h-1h', '1h-6h', '6h-11h', '11h-14h', '14h-17h', '17h-19h', '19h-21h']
    df['time_period'] = 'default'
    for cond, ch in zip(conditions, choices):
        df.loc[cond, 'time_period'] = ch
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
    df.loc[df['zone_code'].isin(['ZONE02']) ,'abnormal_bw'] = df[df['zone_code'].isin(['ZONE02'])].update_time.apply(lambda date: 0.8 if pd.to_datetime(date) in abnormals else 0)
    df.loc[df['zone_code'].isin(['ZONE03']) ,'abnormal_bw'] = df[df['zone_code'].isin(['ZONE03'])].update_time.apply(lambda date: 0.2 if pd.to_datetime(date) in abnormals else 0)

    # For max users
    df.loc[df['zone_code'].isin(['ZONE01']) ,'abnormal_u'] = df[df['zone_code'].isin(['ZONE01'])].update_time.apply(lambda date: -1 if pd.to_datetime(date) in abnormals else 0)
    df.loc[df['zone_code'].isin(['ZONE02']) ,'abnormal_u'] = df[df['zone_code'].isin(['ZONE02'])].update_time.apply(lambda date: 0.8 if pd.to_datetime(date) in abnormals else 0)
    df.loc[df['zone_code'].isin(['ZONE03']) ,'abnormal_u'] = df[df['zone_code'].isin(['ZONE03'])].update_time.apply(lambda date: 0.6 if pd.to_datetime(date) in abnormals else 0)

    # Holidays
    holidays = pd.to_datetime(['2018-01-01', '2017-12-23', '2017-12-24', '2017-12-25',
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

    # Creates time variables
    max_time = df['ds'].max().floor('D')
    delta1m = pd.Timedelta('30 days')
    delta3m = pd.Timedelta('90 days')
    delta6m = pd.Timedelta('180 days')
    delta1y = pd.Timedelta('365 days')

    past_1m = max_time - delta1m
    past_3m = max_time - delta3m
    past_6m = max_time - delta6m
    past_1y = max_time - delta1y

    # Medians from the last 1,3,6,12 months
    zones_1y = df[(df['ds'] >= past_1y) & (df['ds'] <= max_time)].groupby(['zone_code'], as_index=False).agg({
        'max_user': 'median', 
        'bandwidth_total': 'median' 
    })
    zones_1y.columns = ['zone_code','median_user_1y','median_bw_1y']
    zones_1y['median_bw_per_user_1y'] = zones_1y['median_bw_1y'] / zones_1y['median_user_1y']

    zones_1m = df[(df['ds'] >= past_1m) & (df['ds'] <= max_time)].groupby(['zone_code'], as_index=False).agg({
        'max_user': 'median', 
        'bandwidth_total': 'median' 
    })
    zones_1m.columns = ['zone_code','median_user_1m','median_bw_1m']
    zones_1m['median_bw_per_user_1m'] = zones_1m['median_bw_1m'] / zones_1m['median_user_1m']

    zones_3m = df[(df['ds'] >= past_3m) & (df['ds'] <= max_time)].groupby(['zone_code'], as_index=False).agg({
        'max_user': 'median', 
        'bandwidth_total': 'median' 
    })
    zones_3m.columns = ['zone_code','median_user_3m','median_bw_3m']
    zones_3m['median_bw_per_user_3m'] = zones_3m['median_bw_3m'] / zones_3m['median_user_3m']

    zones_6m = df[(df['ds'] >= past_6m) & (df['ds'] <= max_time)].groupby(['zone_code'], as_index=False).agg({
        'max_user': 'median',
        'bandwidth_total': 'median' 
    })
    zones_6m.columns = ['zone_code','median_user_6m','median_bw_6m']
    zones_6m['median_bw_per_user_6m'] = zones_6m['median_bw_6m'] / zones_6m['median_user_6m']

    # Autocorrelation features
    zones_autocorr = df[(df['ds'] >= past_3m) & (df['ds'] <= max_time)].groupby(['zone_code'], as_index=False).agg({
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