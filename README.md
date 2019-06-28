(Vietnamese version [here](README.vi.md))
# AIVIVN's 5th competition: Server bandwidth prediction
My solution for AIVIVN's 5th competition: [Server bandwidth prediction](https://www.aivivn.com/contests/5). To generate the submission file, install the dependencies using the command: `pip install -r requirements.txt`, and then run the `main.py` file.

## Context
A company provides an entertainment platform for music, video, live stream, chat, etc. The company server system is divided into zones by geographic area. In order to meet the growing number of users, the company is interested in forecasting the total bandwidth of each server zone and the maximum number of users simultaneously accessing the server within the next month.

## Data
The data files are located in the `data` directory

### Training set
There are about 35k lines in the `train.csv` file. Here are the first 5 rows.

```csv
UPDATE_TIME,ZONE_CODE,HOUR_ID,BANDWIDTH_TOTAL,MAX_USER
2017-10-01,ZONE01,0,16096.71031272728,212415.0
2017-10-01,ZONE01,1,9374.20790727273,166362.0
2017-10-01,ZONE01,2,5606.225750000003,146370.0
2017-10-01,ZONE01,3,4155.654660909094,141270.0
2017-10-01,ZONE01,4,3253.9785936363623,139689.0
```

- `UPDATE_TIME`: The date
- `HOUR_ID`: The hour
- `ZONE_CODE`: Zone code
- `BANDWIDTH_TOTAL`: Total bandwidth in an hour
- `MAX_USER`: Maximum number of users in an hour (must be natural numbers).

### Testing set
The file `test_id.csv` has 2227 lines. Here are the first few lines:

```csv
id,UPDATE_TIME,ZONE_CODE,HOUR_ID
0,2019-03-10,ZONE01,0
1,2019-03-10,ZONE01,1
2,2019-03-10,ZONE01,2
3,2019-03-10,ZONE01,3
4,2019-03-10,ZONE01,4
5,2019-03-10,ZONE01,5
```
Apart from the above mentioned features, we also have `id` which is the unique id for the datapoints.

## Approach
I used 2 slightly different XGBoost models, one for each of the target variable we have (`bandwidth_total` and `max_user`). The part that took me most of the time is not training, but feature engineering.

### Features
Here are the features I included in the data before feeding it into XGBoost.
#### Time features:
- Hour
- Day
- Week
- Day of week
- Month
- Day of year
- Year

#### Special event features
##### Abnormal events
There are 2 periods in the training data where we see:
1. `ZONE01`: A sudden huge decrease in both total bandwidth and max users.
2. `ZONE02`: A sudden huge increase in both total bandwidth and max users.
3. `ZONE03`: No sudden increase in total bandwidth, a slight increase in max users (compared to `ZONE02`).

And those 2 periods happen to be roughly at the same time of the year (for 2018 and 2019).

From those observations, I decided to create 2 features `abnormal_bw` and `abnormal_u` which show the impact of those days in the target variables. For `ZONE01`, since both bandwidth and max users had a big decrease, I set `abnormal_bw` and `abnormal_u` to `-1`. Similarly for `ZONE02`, these 2 values will be `1` on those days and `0` otherwise. For `ZONE03`, I set `abnormal_bw` to `0.2` since there is only a slight increase in total bandwidth, but `abnormal_u` to `0.8` since the increase was more noticeable.

##### Holiday events
I incorporated the holiday events of Vietnam (in 2018 and 2019) into the boolean feature `holiday`.

#### Zone features
##### Medians
From the training data, I could extract the median total bandwidth and median max users for each zone, with the time window from last 1 month to last 12 months. Here are the list of such features
- `median_user_1m` 
- `median_bw_1m`  
- `median_user_3m` 
- `median_bw_3m` 
- `median_user_6m` 
- `median_bw_6m` 
- `median_user_1y` 
- `median_bw_1y`

##### Autocorrelation features
I also calculated the lag features for 1 day, 3 days and 1 week of the last quarter of the data.

#### Linear regression prediction
Finally, I used the Ridge regression model to fit the training data (using time features, special event features and median features) and use the prediction on both the training and testing data as a feature (for both target variables).
