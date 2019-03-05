# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 16:23:34 2016

@author: charlesmartens
"""

# get sleep continuous df
# get sleep summary stats df
# get hr summary stats df


# ===============================================================
# ***************************************************************
# only working when I use regular spyder on my mac
# if I use anconda5, can’t access fitbit api, something goes wrong
# ***************************************************************
# ===============================================================


cd /Users/charlesmartens/Documents/projects/fitbit_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sys, os
from collections import deque
import fitbit
import configparser
import copy
import statsmodels.formula.api as smf #with this 'formula' api, don't need to create the design #matrices (does it automatically).
from statsmodels.formula.api import *
from matplotlib.dates import DateFormatter
import pickle

sns.set_style('white')



# -----------------------------------------------------------------------------
consumer_key = '2286HS'
consumer_secret = '90a8354c89aa8b6344eb285c87f9e052'

# to get new data, need to use the above consumer_key and consumer_secret and then
# plug them into the website to get a token. so will need to get a new token ea time i download new data
#token = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI0WU1WUDMiLCJhdWQiOiIyMjg2SFMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc2V0IHJhY3QgcmxvYyByaHIgcnBybyByc2xlIiwiZXhwIjoxNTM4Nzc1NTg3LCJpYXQiOjE1MzgxNzA3ODd9.6CW8ZrF0ojxAQO-o-R3jyAnnHx0gHxEgAvr33g7RZCw'
#token = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI0WU1WUDMiLCJhdWQiOiIyMjg2SFMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcmFjdCBybG9jIHJ3ZWkgcmhyIHJudXQgcnBybyByc2xlIiwiZXhwIjoxNTM4Nzk0MDYyLCJpYXQiOjE1MzgxODkyNjJ9.If8hfgK1PRvLyN9XA75cmWcE59IjHNQ8kVeajUftetI'
#token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMjg2SFMiLCJzdWIiOiI0WU1WUDMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJhY3QgcnNldCBybG9jIHJ3ZWkgcmhyIHJwcm8gcm51dCByc2xlIiwiZXhwIjoxNTQzMDk5OTg1LCJpYXQiOjE1NDI0OTUxODV9.4L9BwxqWDmI6nJwioVy80NMOIDoDTGT5v2Nf_FDc8Lc'
#token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMjg2SFMiLCJzdWIiOiI0WU1WUDMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcmFjdCBybG9jIHJ3ZWkgcmhyIHJudXQgcnBybyByc2xlIiwiZXhwIjoxNTQzMDk5OTg1LCJpYXQiOjE1NDI0OTU0OTV9.3tvq4UgNC6olXYerpWQ8O3ezH5ebRcU5xFwjGaq83E0'
#token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMjg2SFMiLCJzdWIiOiI0WU1WUDMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcmFjdCBybG9jIHJ3ZWkgcmhyIHJwcm8gcm51dCByc2xlIiwiZXhwIjoxNTUyMTQ1ODQzLCJpYXQiOjE1NTE1NDEwNDN9.fli8L8INIFeUJERY65BVKAvE29xjrDaElDq7uwHpATI'
token = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMjg2SFMiLCJzdWIiOiI0WU1WUDMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJhY3QgcnNldCBybG9jIHJ3ZWkgcmhyIHJudXQgcnBybyByc2xlIiwiZXhwIjoxNTUyMTQ1ODQzLCJpYXQiOjE1NTE1NDMzMzJ9.7G3YI4bPNCycINNm7LAZ2SgrY7d9Z8TyKkWygPa5i6M'

client = fitbit.FitbitOauth2Client(consumer_key, consumer_secret)
client.authorize_token_url(token)
authd_client = fitbit.Fitbit(client, consumer_secret, oauth2=True, access_token=token)
print(authd_client.API_VERSION)
# set api version to 1.2 so can get sleep stages data
authd_client.API_VERSION=1.2
print(authd_client.API_VERSION)
print(authd_client.API_ENDPOINT)



# ---------------
# test
dates = pd.date_range('06/18/2017', '08/10/2018', freq='D')
date = dates[-1]
date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
stats_sleep = authd_client.sleep(date=date)

dir(authd_client)
dir(authd_client.sleep)
dir(authd_client.activities)
dir(authd_client.activity_stats)

stats_hr_one_day = authd_client.intraday_time_series('activities/heart', base_date=date,
                                             detail_level='1min', start_time='00:00',
                                             end_time='23:59')


# ------------------------------------------------------------------------------
# ----------------------------- get data from api ------------------------------

# --------------------
# get continuous sleep
def pickle_sleep_continuous(authd_client, dates, start, end):
    for date in dates[start:end]:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        try:
            stats_sleep = authd_client.sleep(date=date)
            with open('sleep'+date+'.pkl', 'wb') as picklefile:
                pickle.dump(stats_sleep, picklefile)
        except:
            'no data on ' + date

# full date range - split up into diff segments
# because too many requests to do all at once
#dates = pd.date_range('06/18/2017', '08/10/2018', freq='D')
#dates = pd.date_range('10/10/2016', '11/01/2016', freq='D')
#dates = pd.date_range('11/01/2016', '06/17/2017', freq='D')
#dates = pd.date_range('05/05/2017', '06/14/2017', freq='D')
#dates = pd.date_range('10/01/2016', '10/09/2016', freq='D')

dates = pd.date_range('08/11/2018', '09/25/2018', freq='D')
pickle_sleep_continuous(authd_client, dates, 0, None)

# around 6/18/2017 is when can start getting sleep stages data

def open_sleep_dict(date):
    date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
    with open('sleep'+date+'.pkl', 'rb') as picklefile:
        sleep_data_day_dict = pickle.load(picklefile)
    return sleep_data_day_dict

sleep_day_dict = open_sleep_dict(dates[0])
date = dates[0]



# -----------------
# get continuous hr

def pickle_hr_continuous(authd_client, dates):
    for date in dates:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        try:
            stats_hr = authd_client.intraday_time_series('activities/heart', base_date=date,
                                                         detail_level='1min', start_time='00:00',
                                                         end_time='23:59')
            with open('hr'+date+'.pkl', 'wb') as picklefile:
                pickle.dump(stats_hr, picklefile)
        except:
            'no data on ' + date


#dates = pd.date_range('02/10/2018', '08/10/2018', freq='D')
#dates = pd.date_range('08/10/2017', '02/10/2018', freq='D')
#dates = pd.date_range('10/10/2016', '08/10/2017', freq='D')
#dates = pd.date_range('06/13/2017', '08/10/2017', freq='D')
#dates = pd.date_range('10/01/2016', '10/09/2016', freq='D')
dates = pd.date_range('01/20/2018', '02/09/2018', freq='D')

pickle_hr_continuous(authd_client, dates)

dates = pd.date_range('08/11/2018', '09/25/2018', freq='D')
pickle_hr_continuous(authd_client, dates)


# --------------------
# get activity summary
# (don't need continuous actiity data for now)

GET /1/user/[user-id]/[resource-path]/date/[date]/[period].json
activities/steps

dates = pd.date_range('08/11/2018', '09/25/2018', freq='D')
date = dates[4]
stats_activity = authd_client.intraday_time_series('activities/steps', base_date=date,
                                                   detail_level='1min', start_time='00:00',
                                                   end_time='23:59')

stats_activity = authd_client.intraday_time_series('activities/floors', base_date=date,
                                                   detail_level='1min', start_time='00:00',
                                                   end_time='23:59')

stats_activity.keys()
stats_activity['activities-elevation-intraday'].keys()
stats_activity['activities-elevation-intraday']['dataset']

stats_activity['activities-floors-intraday']['dataset']

l = []
for i in range(len(stats_activity['activities-floors-intraday']['dataset']))[:]:
    l = l + [stats_activity['activities-floors-intraday']['dataset'][i]['value']]

l = []
for i in range(len(stats_activity['activities-elevation-intraday']['dataset']))[:]:
    l = l + [stats_activity['activities-elevation-intraday']['dataset'][i]['value']]

len(l)
sum(l)


stats_activity = authd_client.intraday_time_series('activities/minutesLightlyActive', base_date=date,
                                                   detail_level='1min', start_time='00:00',
                                                   end_time='23:59')

stats_activity = authd_client.intraday_time_series('activities/minutesSedentary', base_date=date,
                                                   detail_level='1min', start_time='00:00',
                                                   end_time='23:59')

# want to be able to know when i'm active. so get below info. i already have the 
# overall summary of steps and elevation. don't need to know when thesea are taking
# place, right? as long as i know whe i'm active v sedentary so can compute resting hr
# so get sedentary first?
#activities/minutesSedentary
#activities/minutesLightlyActive
#activities/minutesFairlyActive
#activities/minutesVeryActive


def pickle_activity(authd_client, dates):
    """Gets activity data. But for many days it doesn't have the times for which
    I was active. But has summary data, e.g., number of steps and elevation gain.
    So can't use this for knowing exactly when I was active vs. not, which would
    be helpeful for computeing resting HR. But can get that info from intraday
    below. And can use this summary data for getting at the impact of activity
    and exercise on my health."""    
    for date in dates:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        try:
            stats_activity = authd_client.activities(date=date)
            with open('activity'+date+'.pkl', 'wb') as picklefile:
                pickle.dump(stats_activity, picklefile)
        except:
            'no data on ' + date

def pickle_sedentary(authd_client, dates):
    """Gets the within-day time series for when i was sedentary (vs. active), 
    coded in the data as 0 or 1 for each minute."""
    for date in dates:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        try:
            stats_activity = authd_client.intraday_time_series('activities/minutesSedentary', base_date=date,
                                                   detail_level='1min', start_time='00:00',
                                                   end_time='23:59')
            with open('sedentary'+date+'.pkl', 'wb') as picklefile:
                pickle.dump(stats_activity, picklefile)
        except:
            'no data on ' + date

def pickle_steps(authd_client, dates):
    """ """
    for date in dates:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        try:
            stats_activity = authd_client.intraday_time_series('activities/steps', base_date=date,
                                                   detail_level='1min', start_time='00:00',
                                                   end_time='23:59')
            with open('steps'+date+'.pkl', 'wb') as picklefile:
                pickle.dump(stats_activity, picklefile)
        except:
            'no data on ' + date

def pickle_elevation(authd_client, dates):
    """ """
    for date in dates:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        try:
            stats_activity = authd_client.intraday_time_series('activities/elevation', base_date=date,
                                                   detail_level='1min', start_time='00:00',
                                                   end_time='23:59')
            with open('elevation'+date+'.pkl', 'wb') as picklefile:
                pickle.dump(stats_activity, picklefile)
        except:
            'no data on ' + date

def pickle_floors(authd_client, dates):
    """ """
    for date in dates:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        try:
            stats_activity = authd_client.intraday_time_series('activities/floors', base_date=date,
                                                   detail_level='1min', start_time='00:00',
                                                   end_time='23:59')
            with open('floors'+date+'.pkl', 'wb') as picklefile:
                pickle.dump(stats_activity, picklefile)
        except:
            'no data on ' + date


#dates = pd.date_range('08/10/2017', '08/10/2018', freq='D')
#dates = pd.date_range('01/20/2018', '08/10/2018', freq='D')
#dates = pd.date_range('07/20/2018', '08/10/2018', freq='D')

#dates = pd.date_range('10/01/2016', '01/28/2017', freq='D')
#dates = pd.date_range('01/28/2017', '01/10/2018', freq='D')  
#dates = pd.date_range('01/10/2018', '08/10/2018', freq='D')  
#dates = pd.date_range('08/10/2018', '09/26/2018', freq='D')  # pause for a while and continue
#dates = pd.date_range('07/24/2017', '12/31/2017', freq='D')  # pause for a while and continue
#dates = pd.date_range('2016-12-12', '2017-05-07', freq='D')


# try getting floors for this date range again. all zeros.
dates = pd.date_range('2016-12-12', '2017-05-07', freq='D')
# test first:
dates = pd.date_range('2016-12-12', '2016-12-15', freq='D')



pickle_activity(authd_client, dates)

pickle_sedentary(authd_client, dates)

pickle_steps(authd_client, dates)

pickle_elevation(authd_client, dates)
# elevation not correct

pickle_floors(authd_client, dates)
# finished - pickled up to 9-26




# explore
date_list = pd.date_range('2016-10-01', '2018-09-25', freq='D')
date_list = date_list[100:105]

date_list = pd.date_range('2019-02-25', '2019-02-27', freq='D')


def open_metric_dict(date, metric):
    date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
    try:
        with open(metric+date+'.pkl', 'rb') as picklefile:
            activity_data_day_dict = pickle.load(picklefile)
    except:
        print(date)
        'no dict on ' + date
        print()
        activity_data_day_dict = None
    return activity_data_day_dict

date = date_list[1]
elevation_data_day_dict = open_metric_dict(date, 'elevation')
elevation_data_day_dict.keys()
elevation_data_day_dict['activities-elevation']
elevation_data_day_dict['activities-elevation-intraday'].keys()
elevation_data_day_dict['activities-elevation-intraday']['dataset']

metric='elevation'

def create_df_for_metric_each_minute(date_list, metric):
    time_and_steps_dict = {'date':[], 'time':[], metric:[]}
    for date in date_list:
        steps_data_day_dict = open_metric_dict(date, metric)
        if steps_data_day_dict == None:
            None
        else:
            dataset_day = steps_data_day_dict['activities-'+metric+'-intraday']['dataset']
            for data_dict in dataset_day:
                time_and_steps_dict['date'].append(date)
                time_and_steps_dict['time'].append(data_dict['time'])
                time_and_steps_dict[metric].append(data_dict['value'])
    df_time_steps = pd.DataFrame(time_and_steps_dict)
    return df_time_steps  


dates = pd.date_range('01/01/2018', '09/01/2018', freq='D')

df_time_floors = create_df_for_metric_each_minute(dates, 'floors')
df_time_floors.groupby('date')['floors'].sum().hist(bins=15, alpha=.6)
plt.grid(False)

df_time_elevation = create_df_for_metric_each_minute(dates, 'elevation')
df_time_elevation.groupby('date')['elevation'].sum().hist(bins=15, alpha=.6)
plt.grid(False)

# returning zeros

df_time_elevation = create_df_for_metric_each_minute(dates, 'elevation')
df_time_elevation.groupby('date')['elevation'].sum().hist(bins=15, alpha=.6)
plt.grid(False)


pickle_elevation(authd_client, date_list)
df_time_elevation = create_df_for_metric_each_minute(date_list, 'elevation')

















