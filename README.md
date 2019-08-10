(Vietnamese version [here](README.vi.md))
# AIviVN's 5th competition: Server bandwidth prediction
My solution to AIviVN's 5th competition: [Server bandwidth prediction](https://www.aivivn.com/contests/5). 

## How to generate the submission file
To generate the submission file (tested on Python 3.6 & Python 3.7): 
- Download the `csv` data files (`test_id.csv` and `train.csv`), put them into a folder and name it `data`.
- Install the dependencies if needed using the command: `pip install -r requirements.txt`.
- Run the `main_combined.py` file.

## Context
*(Taken from the competition's webpage, translated to English)*

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
The final prediction is the weighted average of XGBoost predictions and the predictions using median of medians(`windows = [1,2]`): `final_prediction = 0.8 * XGBoost + 0.2 * median_of_medians`. As for XGBoost, I used 2 identical models for both target variables (`bandwidth_total` and `max_user`). The part that took me most of the time is feature engineering. 

### Filling missing values
There are some missing values in the time series for all zones. My strategy is to use the value of the same day in the previous week to fill the nulls.

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

From those observations, I decided to create 2 features `abnormal_bw` and `abnormal_u` which show the impact of those days in the target variables. 
- For `ZONE01`, since both bandwidth and max users had a big decrease, I set `abnormal_bw` and `abnormal_u` to `-1`. 
- Similarly for `ZONE02`, these 2 values will be `0.8` on such days and `0` otherwise. 
- For `ZONE03`, I set `abnormal_bw` to `0.2` since there is only a slight increase in total bandwidth, but `abnormal_u` to `0.6` since the increase was more noticeable.

##### Holiday events
I incorporated the holiday events of Vietnam (in 2018 and 2019) into the boolean feature `holiday`. I also added the Christmas days in 2017 (from Dec 23 to Dec 25) since the total bandwidth and max users in `ZONE01` had suddenly increased during those days

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
- `median_bw_per_user_6m`
- `median_bw_per_user_3m`
- `median_bw_per_user_1m`
- `median_bw_per_user_1y`

##### Autocorrelation features
I also calculated the lag features for 1 day, 3 days and 1 week of the last quarter of the data.

#### Linear regression prediction features
Finally, I used the Ridge regression model to fit the training data (using time features, special event features and median features) and use the prediction on both the training and testing data as a feature (for both target variables). 

## Results
The final model gives `sMAPE = 5.12708` on the public LB and `sMAPE = 5.18001` on the final LB.

## Future works
There are a lot of room for improvement in this problem. With the enormous development of deep learning research in the recent years, one can try incorporating deep neural network models into this problem and see if it could surpass traditional ML approaches, which rely very much on the features of the dataset.

## References
- [1] [Time series Introduction](https://people.maths.bris.ac.uk/~magpn/Research/LSTS/STSIntro.html)
- [2] [XGBoost Mathematics Explained](https://towardsdatascience.com/xgboost-mathematics-explained-58262530904a)
- [3] [Using XGBoost in Python](https://www.datacamp.com/community/tutorials/xgboost-in-python)
- [4] [Basic time series manipulation with pandas](https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea)
- [5] [Tutorial: Time series forecasting with XGBoost](https://www.kaggle.com/robikscube/tutorial-time-series-forecasting-with-xgboost)
