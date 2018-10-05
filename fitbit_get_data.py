# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 16:23:34 2016

@author: charlesmartens
"""

# get sleep continuous df
# get sleep summary stats df
# get hr summary stats df

#only working when I use regular spyder on my mac
#if I use anconda5, can’t access fitbit api, something goes wrong

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
token = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI0WU1WUDMiLCJhdWQiOiIyMjg2SFMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcmFjdCBybG9jIHJ3ZWkgcmhyIHJudXQgcnBybyByc2xlIiwiZXhwIjoxNTM4Nzk0MDYyLCJpYXQiOjE1MzgyNTE5MTJ9.Fjkul07xFarNcv7TCT-IqCnfr9S7L1Crh8q2FAMd_zo'

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

def pickle_activity(authd_client, dates):
    for date in dates:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        try:
            stats_activity = authd_client.activities(date=date)
            with open('activity'+date+'.pkl', 'wb') as picklefile:
                pickle.dump(stats_activity, picklefile)
        except:
            'no data on ' + date


#dates = pd.date_range('08/10/2017', '08/10/2018', freq='D')
#dates = pd.date_range('01/20/2018', '08/10/2018', freq='D')
#dates = pd.date_range('07/20/2018', '08/10/2018', freq='D')
#dates = pd.date_range('10/10/2016', '08/10/2017', freq='D')
#dates = pd.date_range('04/14/2017', '08/10/2017', freq='D')
#dates = pd.date_range('10/01/2016', '10/09/2016', freq='D')
dates = pd.date_range('08/11/2018', '09/25/2018', freq='D')

pickle_activity(authd_client, dates)
