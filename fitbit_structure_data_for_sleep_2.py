# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 16:23:34 2016

@author: charlesmartens

Analyses:
Alcohol on HR while asleep (presented at Google).

"""

# get new sleep quality metrics
# start on L 2210

## ===============================================================
## save df with sedentary hr mean ea day as measure of resting hr
## computed by getting the mean sedentary hr each hour and then taking
## the average of these hourly measurements
# df_sedentary_hr_mean_day.to_pickle('df_sedentary_hr_mean_day.pkl')
## save df with minimum sedentary hr mean ea day as measure of resting hr
## computed by getting the minium sedentary hr each hour and then taking
## the average of these hourly measurements
# df_minimum_hr_mean_day.to_pickle('df_minimum_hr_mean_day.pkl')
## ===============================================================


#cd \\chgoldfs\253Broadway\PrivateFiles\amartens\hr\hr_sleep_data\fitbit_data2
cd /Users/charlesmartens/Documents/projects/fitbit_data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import sys, os
from collections import deque
import configparser
import copy
import statsmodels.formula.api as smf #with this 'formula' api, don't need to create the design #matrices (does it automatically).
from statsmodels.formula.api import *
from matplotlib.dates import DateFormatter
import pickle
from datetime import datetime, timedelta
#from datetime import datetime
#from datetime import timedelta
#from scipy import fft, arange
sns.set_style('white')



# -----------------------------------------------------------------------------
# ---------------------- organized saved pickle files -------------------------

# -----
# sleep

def open_sleep_dict(date):
    date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
    try:
        with open('sleep'+date+'.pkl', 'rb') as picklefile:
            sleep_data_day_dict = pickle.load(picklefile)
    except:
        print()
        'no dict on ' + date
        print()
    return sleep_data_day_dict

def get_sleep_summary_for_all_sleep_in_dict(sleep_data_day_dict):
    """ If more than 1 period of sleep, then have multiple sleep records
        and the following summarizes all the sleep periods for that day. """
    sleep_records_number_day = sleep_data_day_dict['summary']['totalSleepRecords']
    minutes_sleep_day = sleep_data_day_dict['summary']['totalMinutesAsleep']
    if 'stages' in sleep_data_day_dict['summary'].keys():
        summary_stages_day = sleep_data_day_dict['summary']['stages']
    else:
        summary_stages_day = np.nan
    time_in_bed_day = sleep_data_day_dict['summary']['totalTimeInBed']
    return sleep_records_number_day, minutes_sleep_day, summary_stages_day, time_in_bed_day
    
def get_sleep_info_for_day_dicts(date):
    sleep_data_day_dict = open_sleep_dict(date)
    sleep_record_classic_or_stages = sleep_data_day_dict['sleep'][0]['type'] 
    sleep_wakings_timeline_day = []
    start_end_sleep_timeline_day = []
    for sleep_record in range(len(sleep_data_day_dict['sleep'])):
        sleep_wakings_timeline = sleep_data_day_dict['sleep'][sleep_record]['levels']['data']  
        end_sleep = sleep_data_day_dict['sleep'][sleep_record]['endTime']  
        start_sleep = sleep_data_day_dict['sleep'][sleep_record]['startTime']      
        start_end_sleep_timeline = [(start_sleep, end_sleep)]
        #sleep_wakings_timeline = [{'dateTime':start_sleep, 'level':'start_sleep', 'seconds':np.nan}] + sleep_wakings_timeline
        #sleep_wakings_timeline = sleep_wakings_timeline + [{'dateTime':end_sleep, 'level':'end_sleep', 'seconds':np.nan}]
        sleep_wakings_timeline_day = sleep_wakings_timeline_day + sleep_wakings_timeline
        start_end_sleep_timeline_day = start_end_sleep_timeline_day + start_end_sleep_timeline
    sleep_records_number_day, minutes_sleep_day, summary_stages_day, time_in_bed_day = get_sleep_summary_for_all_sleep_in_dict(sleep_data_day_dict)
    return sleep_record_classic_or_stages, sleep_wakings_timeline_day, start_end_sleep_timeline_day, sleep_records_number_day, minutes_sleep_day, summary_stages_day, time_in_bed_day

def return_dicts_w_sleep_info_for_list_of_dates(dates):
    date_to_classic_or_stages_dict = {}
    date_to_sleeps_dict = {}
    date_to_minutes_asleep_dict = {}
    # don't believe these stages dicts but keep for now
    # they're freq exactly the same for diff dates
    # also not sure date of date_to_classic_or_stages_dict and 
    # date_to_minutes_asleep_dict matches up with date as I'll be thinking about it
    date_to_light_dict = {}
    date_to_deep_dict = {}
    date_to_rem_dict = {}
    date_to_wake_dict = {}
    #sleep_stages_timeline_all = []
    sleep_type_events_timeline_all = []
    start_end_sleep_timeline_all = []
    for date in dates:
        try:
            sleep_data_day_dict = open_sleep_dict(date)
            print(date)
            sleep_record_classic_or_stages, sleep_wakings_timeline_day, start_end_sleep_timeline_day, sleep_records_number_day, minutes_sleep_day, summary_stages_day, time_in_bed_day = get_sleep_info_for_day_dicts(date)
            sleep_type_events_timeline_all = sleep_type_events_timeline_all + sleep_wakings_timeline_day
            start_end_sleep_timeline_all = start_end_sleep_timeline_all + start_end_sleep_timeline_day
            date_to_classic_or_stages_dict[date] = sleep_record_classic_or_stages
            date_to_sleeps_dict[date] = sleep_records_number_day
            date_to_minutes_asleep_dict[date] = minutes_sleep_day
            date_to_light_dict[date] = summary_stages_day['light']
            date_to_deep_dict[date] = summary_stages_day['deep']
            date_to_rem_dict[date] = summary_stages_day['rem']
            date_to_wake_dict[date] = summary_stages_day['wake']
        except:
            print()
            #print('no dict on ' + str(date))
            print()
    return sleep_type_events_timeline_all, start_end_sleep_timeline_all, date_to_classic_or_stages_dict, date_to_sleeps_dict, date_to_minutes_asleep_dict

def get_df_sleep_30_sec_records(start_end_sleep_timeline_all):
    asleep_times_all = []
    for row in start_end_sleep_timeline_all:
        print(row)
        #dates_sleep_episode = pd.date_range(row[0], row[1], freq='30s')
        dates_sleep_episode = list(pd.date_range(row[0], row[1], freq='30s'))
        asleep_times_all = asleep_times_all + dates_sleep_episode
    df_asleep_times = pd.DataFrame(asleep_times_all).rename(columns={0:'date_time'})
    return df_asleep_times

def get_dict_for_30_sec_records_for_diff_sleep_levels(sleep_type_events_timeline_all):
    """ Takes the sleep_type_events_timeline_all with a row for a type of sleep event
    and the number of seconds of that event. Produces a dict with the 30-sec 
    periods was in each of the sleep categories in the dictionary below. """
    sleep_info_dict = {'deep':[], 'light':[], 'rem':[], 'wake':[], 
                       'restless':[], 'asleep':[], 'awake':[], 'unknown':[]}
    for sleep_info_row in sleep_type_events_timeline_all[:]:
        print(sleep_info_row)
        sleep_info = sleep_info_row['level']
        time = sleep_info_row['dateTime']
        periods_to_enter = sleep_info_row['seconds']/30
        times_in_sleep_level = list(pd.date_range(start=time, periods=periods_to_enter, freq='30s'))
        sleep_info_dict[sleep_info] = sleep_info_dict[sleep_info] + times_in_sleep_level
    return sleep_info_dict  


def map_sleep_event_types_onto_df_sleep(df_sleep, sleep_info_dict, sleep_event):
    df_sleep_event = pd.DataFrame(sleep_info_dict[sleep_event]).rename(columns={0:'date_time'})
    df_sleep_event[sleep_event] = 1
    datetime_to_sleep_event_dict = dict(zip(df_sleep_event['date_time'], df_sleep_event[sleep_event]))
    df_sleep[sleep_event] = df_sleep['date_time'].map(datetime_to_sleep_event_dict)
    df_sleep[sleep_event].replace(np.nan, 0, inplace=True)
    return df_sleep

def map_start_and_end_sleep_times_onto_df_sleep(df_sleep, start_end_sleep_timeline_all):
    df_start_end_times = pd.DataFrame(start_end_sleep_timeline_all).rename(columns={0:'start', 1:'end'})
    df_start_end_times['flag'] = 1
    df_start_end_times['start'] = pd.to_datetime(df_start_end_times['start'])
    df_start_end_times['end'] = pd.to_datetime(df_start_end_times['end'])
    datetime_to_start_sleep_time_dict = dict(zip(df_start_end_times['start'], df_start_end_times['flag']))
    datetime_to_end_sleep_time_dict = dict(zip(df_start_end_times['end'], df_start_end_times['flag']))
    df_sleep['start_sleep'] = df_sleep['date_time'].map(datetime_to_start_sleep_time_dict)
    df_sleep['end_sleep'] = df_sleep['date_time'].map(datetime_to_end_sleep_time_dict)
    return df_sleep

def consolidate_wake_and_awake_flags(df_sleep):
    df_sleep.loc[(df_sleep['awake']==1) | 
            (df_sleep['wake']==1), 'awake'] = 1
    del df_sleep['wake']
    return df_sleep

# don't really need this f now, but can use for double check
#def map_on_sleep_wake_info(df_sleep, sleep_type_events_timeline_all):
#    """Takes df_sleep -- with a row for each 30-sec period that asleep -- and
#    maps on whether that period was the beginning of a sleep type (e.g., rem, deep, awake)."""
#    df_sleep_wake = pd.DataFrame(sleep_type_events_timeline_all)
#    df_sleep_wake['date_time'] = pd.to_datetime(df_sleep_wake['dateTime'])
#    del df_sleep_wake['dateTime']
#    datetime_to_level_sleep_dict = dict(zip(df_sleep_wake['date_time'], df_sleep_wake['level']))
#    datetime_to_seconds_at_level_sleep_dict = dict(zip(df_sleep_wake['date_time'], df_sleep_wake['seconds']))
#    df_sleep['level_sleep'] = df_sleep['date_time'].map(datetime_to_level_sleep_dict)
#    df_sleep['level_sleep_seconds'] = df_sleep['date_time'].map(datetime_to_seconds_at_level_sleep_dict)
#    return df_sleep


# test
#dates = pd.date_range('2016-10-01', '2016-10-03', freq='D')
#sleep_data_day_dict = open_sleep_dict(dates[1])

# get all dates:
dates = pd.date_range('2016-10-01', '2018-09-25', freq='D')

sleep_type_events_timeline_all, start_end_sleep_timeline_all, date_to_classic_or_stages_dict, date_to_sleeps_dict, date_to_minutes_asleep_dict = return_dicts_w_sleep_info_for_list_of_dates(dates)
df_sleep = get_df_sleep_30_sec_records(start_end_sleep_timeline_all)
sleep_info_dict = get_dict_for_30_sec_records_for_diff_sleep_levels(sleep_type_events_timeline_all)
df_sleep = map_sleep_event_types_onto_df_sleep(df_sleep, sleep_info_dict, 'deep')
df_sleep = map_sleep_event_types_onto_df_sleep(df_sleep, sleep_info_dict, 'light')
df_sleep = map_sleep_event_types_onto_df_sleep(df_sleep, sleep_info_dict, 'rem')
df_sleep = map_sleep_event_types_onto_df_sleep(df_sleep, sleep_info_dict, 'wake')
df_sleep = map_sleep_event_types_onto_df_sleep(df_sleep, sleep_info_dict, 'restless')
df_sleep = map_sleep_event_types_onto_df_sleep(df_sleep, sleep_info_dict, 'awake')

df_sleep.shape  # (725079, 7)
df_sleep.head(10)
#df_sleep = map_sleep_event_types_onto_df_sleep(df_sleep, sleep_info_dict, 'unknown')
#sleep_info_dict['unknown']

df_sleep = consolidate_wake_and_awake_flags(df_sleep)

# examine pct of sleep records with each type of sleet event-state
sleep_event_types_list = ['deep', 'light', 'rem', 'restless', 'awake']
for sleep_event_type in sleep_event_types_list:
    print(sleep_event_type)
    print(df_sleep[sleep_event_type].value_counts(normalize=True).round(3))
    print()

#df_sleep['sum_sleep_event_types'] = df_sleep[sleep_event_types_list].sum(axis=1)
#df_sleep['sum_sleep_event_types'].value_counts()
# no rows have two event types. good.
# some rows have no event types. count those as sleep?

# map on one more col that has start and end of sleep periods
df_sleep = map_start_and_end_sleep_times_onto_df_sleep(df_sleep, start_end_sleep_timeline_all)
df_sleep.head()
df_sleep.tail()

# save sleep df
df_sleep.to_pickle('df_sleep.pkl')
df_sleep[['date_time']].to_pickle('df_sleep_skeleton.pkl')

# descriptives
# loop though -- by day, how many hours alseep for?

# get df_awake 
# get full df of 30-sec periods and delete df_sleep?
# then map on hr. then compute resting hr w 10-min moving avg.

# map on hr to sleep
# create sleep metrics? 
# plot several sleep hr timeseries in a row to think about what to extract?


# -------
# hr data

def open_hr_dict(date):
    date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
    try:
        with open('hr'+date+'.pkl', 'rb') as picklefile:
            hr_data_day_dict = pickle.load(picklefile)
    except:
        print()
        'no dict on ' + date
        print()
    return hr_data_day_dict


dates = pd.date_range('2018-09-20', '2018-09-21', freq='D')
date = dates[0]
hr_data_day_dict = open_hr_dict(date)

hr_data_day_dict.keys()  # dict_keys(['activities-heart-intraday', 'activities-heart'])
hr_data_day_dict['activities-heart-intraday'].keys()  # hr min by min
hr_data_day_dict['activities-heart-intraday']['datasetType']
hr_data_day_dict['activities-heart-intraday']['datasetInterval']
hr_data_day_dict['activities-heart-intraday']['dataset'][:30]
hr_data_day_dict['activities-heart-intraday']['dataset'][-30:]

hr_data_day_dict['activities-heart']  # hr zone stuff: fat burn and cardio


def produce_df_hr(dates):
    """ Give dates and produce df hr. """
    hr_ts_list = []
    date_ts_list = []
    for date in dates:
        date_short = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date_short)
        try:
            hr_data_day_dict = open_hr_dict(date)
            hr_ts_list = hr_ts_list + hr_data_day_dict['activities-heart-intraday']['dataset']
            date_list = [date_short]*len(hr_data_day_dict['activities-heart-intraday']['dataset'])
            date_ts_list = date_ts_list + date_list
        except:
            print()
            print('no dict on ' + str(date))
            print()
    df_hr = pd.DataFrame(hr_ts_list).rename(columns={'value':'hr'})
    df_date = pd.DataFrame(date_ts_list).rename(columns={0:'date'})
    df_hr_date = pd.concat([df_date, df_hr], axis=1)
    df_hr_date['date_time'] = pd.to_datetime(df_hr_date['date'] + ' ' +df_hr_date['time'])
    return df_hr_date
 
    
# get all dates:
dates = pd.date_range('2016-10-01', '2018-09-25', freq='D')
df_hr = produce_df_hr(dates)

# save hr df
df_hr.to_pickle('df_hr.pkl')


# ================================
# ================================
# start here
df_hr = pd.read_pickle('df_hr.pkl')


df_hr.shape  # (944712, 4)
df_hr.dtypes
df_hr.head()
df_hr.tail()
 
# pull up df sleep skeleton
# remove those dates from this df hr
# to leave with df hr when awake
# so can compute resting hr
# and can map these hr values onto sleep
# and can fill in 30 sec intervals with interpolate?

df_sleep_dates = pd.read_pickle('df_sleep_skeleton.pkl')
sleep_datetime_list = df_sleep_dates['date_time'].values
len(sleep_datetime_list)

df_hr_awake = df_hr[r-df_hr['date_time'].isin(sleep_datetime_list)]
df_hr_sleep = df_hr[df_hr['date_time'].isin(sleep_datetime_list)]

df_hr_awake.shape  # (599287, 4)
df_hr_sleep.shape  # (345425, 4)

#df_hr_awake.to_pickle('df_hr_awake.pkl')
#df_hr_sleep.to_pickle('df_hr_sleep.pkl')

df_hr_awake.head()

# distribution of times w hr
sns.countplot(df_hr['date_time'].dt.hour, color='green', alpha=.4)
plt.xlabel('Hour', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.despine()

# distribution of times asleep w hr
sns.countplot(df_hr_awake['date_time'].dt.hour, color='dodgerblue', alpha=.5)
plt.xlabel('Hour', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
sns.despine()

# distribution of times awake w hr
sns.countplot(df_hr_sleep['date_time'].dt.hour, color='orange', alpha=.5)
plt.xlabel('Hour', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
sns.despine()



# ------------------------------
# sleep focus and hr while sleep
df_hr_sleep.head()
date_time_to_hr_dict = dict(zip(df_hr_sleep['date_time'], df_hr_sleep['hr']))

df_sleep = pd.read_pickle('df_sleep.pkl')
df_sleep.head()
df_sleep.tail()
# these are in 30-sec increments. df_hr_sleep in 1 min increments
# could map on hr_sleep and interpolate or get mean hr of before and after

# give new date that relates to the day the sleep started. 
# e.g., if tue is july 1, then that night is sleep associated with july 1
# to do this, substract 12 hours from the date. so that 11:59am that morning
# becomes 11:49pm of the prior day. and the earlist time, 18:00 (6pm) would
# become 6am. so now that whole time frame is labeled as one date, teh date
# when it started. and then just leave the date
df_sleep['date_sleep'] = df_sleep['date_time'] - timedelta(hours=12)
df_sleep['date_sleep'] = pd.to_datetime(df_sleep['date_sleep'].dt.date)
df_sleep.head()


# ----------------------------------------
# get alc rating days in here and truncate

# import daily ratings
# get daily questions -
df_daily_qs_early = pd.read_excel('Mood Measure (Responses).xlsx')
df_daily_qs_early.head()

df_daily_qs_early = df_daily_qs_early[df_daily_qs_early['What are your initials']=='AM']
df_daily_qs_early['date'] = pd.to_datetime(df_daily_qs_early['Timestamp'].dt.year.astype(str) + '-' + df_daily_qs_early['Timestamp'].dt.month.astype(str) + '-' + df_daily_qs_early['Timestamp'].dt.day.astype(str))
df_daily_qs_early['date_short'] = df_daily_qs_early['date'].dt.date
df_daily_qs_early.head()

# qs to use and merge with current qs?
df_daily_qs_early.dtypes
df_daily_qs_early = df_daily_qs_early[['date', 'Today, are you tired?', 'Last night, did you sleep well?']]
df_daily_qs_early['energy'] = 5 - df_daily_qs_early['Today, are you tired?']
df_daily_qs_early.rename(columns={'Last night, did you sleep well?':'sleep'}, inplace=True)
df_daily_qs_early = df_daily_qs_early[['date', 'energy', 'sleep']]


df_daily_qs_current = pd.read_excel('Daily_Measure_2018_10_8.xlsx')
#df_daily_qs_current = pd.read_csv('Daily_Measure_from_2_26_17.csv')
df_daily_qs_current.head()
df_daily_qs_current.tail()

df_daily_qs_current['Timestamp'] = pd.to_datetime(df_daily_qs_current['Timestamp'])
df_daily_qs_current['year'] = df_daily_qs_current['Timestamp'].dt.year.astype(str).str.split('.').str[0]
df_daily_qs_current['month'] = df_daily_qs_current['Timestamp'].dt.month.astype(str).str.split('.').str[0]
df_daily_qs_current['day'] = df_daily_qs_current['Timestamp'].dt.day.astype(str).str.split('.').str[0]
df_daily_qs_current['date'] = df_daily_qs_current['year'] + '-' + df_daily_qs_current['month'] + '-' + df_daily_qs_current['day']
df_daily_qs_current['date'].replace('nan-nan-nan', np.nan, inplace=True)
df_daily_qs_current['date'] = pd.to_datetime(df_daily_qs_current['date'])
df_daily_qs_current.columns

df_daily_qs_current = df_daily_qs_current[['date', 'Right now I feel energetic.',
       'At some point today I felt annoyed.', 'At some point today I had fun with someone.',
       'Last night, I woke up from a restorative sleep.', 'Last night I had ____ alcoholic drinks',
       'The alcohol I drank last night was ____', 'Notes', 'Right now I feel tired.', 
       'At some point today I felt insulted.']]

df_daily_qs_current.columns
df_daily_qs_current.columns = ['date', 'energy', 'annoyed', 'fun', 'sleep', 'alcohol', 'alcohol_type', 'notes', 'tired', 'insulted']

for col in df_daily_qs_current.columns:
    print(col, len(df_daily_qs_current[df_daily_qs_current[col].isnull()]))

for col in df_daily_qs_current.columns:
    print(col, len(df_daily_qs_current[df_daily_qs_current[col].notnull()]))
# given that i have annoyed and energy for just about all rows, i can delete tired and insulted
df_daily_qs_current = df_daily_qs_current[['date', 'energy', 'annoyed', 'fun', 
                                           'sleep', 'alcohol', 'alcohol_type',
                                           'notes']]

df_daily_qs_current['alcohol']
df_daily_qs_current[df_daily_qs_current['alcohol'].notnull()]

df_daily_qs = pd.concat([df_daily_qs_early, df_daily_qs_current], ignore_index=True)
df_daily_qs.head(20)

# 6 duplicate dates
len(df_daily_qs['date'].unique())
len(df_daily_qs)

dates_duplicated_list = df_daily_qs[df_daily_qs['date'].duplicated()]['date'].values
df_daily_qs[df_daily_qs['date'].isin(dates_duplicated_list)]
df_daily_qs = df_daily_qs.drop_duplicates(subset='date', keep='last')
df_daily_qs.tail()

# the alcohol from the date on df_daily_qs refers to previous date
df_daily_qs['date_prior'] = df_daily_qs['date'] - timedelta(days=1)
df_daily_qs['date_two_prior'] = df_daily_qs['date'] - timedelta(days=2)
df_daily_qs[['date_prior', 'date_two_prior', 'date']]

def create_lagged_date_variables(number_of_lags, date_variable, df):
    for lag in list(range(1,number_of_lags+1)):
        df['date_lag_'+str(lag)] = df[date_variable] - timedelta(days=lag)
    return df

df_daily_qs = create_lagged_date_variables(2, 'date', df_daily_qs)
df_daily_qs.head()

df_daily_qs[['date_lag_1', 'alcohol']]
df_daily_qs_w_alcohol = df_daily_qs[df_daily_qs['alcohol'].notnull()]
df_daily_qs_w_alcohol = df_daily_qs_w_alcohol.reset_index(drop=True)
df_daily_qs_w_alcohol[['date_lag_1', 'alcohol']]
df_daily_qs_w_alcohol[['date_lag_1', 'alcohol']][:50]

sns.relplot(x='date_lag_1', y='alcohol', data=df_daily_qs_w_alcohol, kind='line')

df_daily_qs_w_alcohol_from_6_1_17 = df_daily_qs_w_alcohol[df_daily_qs_w_alcohol['date_lag_1']>='2017-09-01']
df_daily_qs_w_alcohol_from_6_1_17[['date_lag_1', 'alcohol']]

# alt date to alcohol dict starting a bit later
#date_to_alcohol_dict = dict(zip(df_daily_qs_w_alcohol_from_6_1_17['date_prior'], df_daily_qs_w_alcohol_from_6_1_17['alcohol']))
date_to_alcohol_dict = dict(zip(df_daily_qs['date_lag_1'], df_daily_qs['alcohol']))
df_sleep['alcohol'] = df_sleep['date_sleep'].map(date_to_alcohol_dict)
len(df_sleep)  # 725079
len(df_sleep['date_sleep'].unique())  # 708
len(df_sleep[df_sleep['alcohol'].notnull()])  # 708
df_test = df_sleep.groupby('date_sleep')['alcohol'].mean()
len(df_test[df_test.notnull()])  # 480 days w alcohol ratings

df_daily_qs['alcohol_type'].unique()
date_to_alcohol_type_dict = dict(zip(df_daily_qs['date_lag_1'], df_daily_qs['alcohol_type']))

# truncate so only alcohol days
df_sleep = df_sleep[df_sleep['alcohol'].notnull()]
len(df_sleep)  # 486332
len(df_sleep['date_sleep'].unique())  # 480 nights of sleep to anys for alcohol presentation

# vietnam dates
len(df_sleep['date_sleep'].unique())  # 480
dates_vietnam_list = pd.date_range('2016-11-02', '2016-11-12')
df_sleep = df_sleep[-df_sleep['date_sleep'].isin(dates_vietnam_list)]
len(df_sleep['date_sleep'].unique())  # 480

# remove vietnam dates - nov 3-12
#len(df_sleep_8_to_11_resampled['date_sleep'].unique())  # 699
#dates_vietnam_list = pd.date_range('2016-11-02', '2016-11-12')
#df_sleep_8_to_11_resampled = df_sleep_8_to_11_resampled[-df_sleep_8_to_11_resampled['date_sleep'].isin(dates_vietnam_list)]
#len(df_sleep_8_to_11_resampled['date_sleep'].unique())  # 697

# plot again but plot the data from the sleep api
plt.figure(figsize=(14, 6), dpi=80)
sns.countplot(df_sleep['date_time'].dt.hour, color='dodgerblue', alpha=.5)
plt.xlabel('Hour of the Day', fontsize=26)
plt.ylabel('Sleep Measurements', fontsize=26)
plt.xticks(fontsize=22)
plt.yticks([],[], fontsize=22)
sns.despine()

df_sleep['hr'] = df_sleep['date_time'].map(date_time_to_hr_dict)

df_sleep.head()
df_sleep_min = df_sleep[df_sleep['date_time'].astype(str).str[-2:-1]=='0']
len(df_sleep)
len(df_sleep_min['date_sleep'].unique())
df_sleep_by_day = df_sleep_min.groupby('date_sleep').size()/60
df_sleep_by_day.mean()  # 8.439036 -- can't be???
df_sleep_min.head()
1 - len(df_sleep_min[df_sleep_min['hr'].isnull()]) / len(df_sleep_min)

len(df_sleep[df_sleep['hr'].isnull()]) / len(df_sleep)
# should be .50. so a few are missing and not sure why
# try a ffill and see what's still null
#df_sleep['hr'].fillna(method='ffill', limit=1, inplace=True)
df_sleep[df_sleep['hr'].isnull()]
# 34370 rows empty still. what happened?
df_missing_hr = df_sleep[df_sleep['hr'].isnull()]
df_missing_hr.head()
df_missing_hr['date_time'].hist()
df_missing_hr['hour'] = df_missing_hr['date_time'].dt.hour
df_missing_hr['hour'].hist()

df_missing_hr['deep'].value_counts(normalize=True)
df_missing_hr['light'].value_counts(normalize=True)
df_missing_hr['rem'].value_counts(normalize=True)
df_missing_hr['restless'].value_counts(normalize=True)
df_missing_hr['restless'].value_counts(normalize=True)
df_missing_hr['awake'].value_counts(normalize=True)

# are there days with tons of hr missing?
# do i first want to just select 8pm to 11:59am. 
# then label these chunks as a particular date.
# then run stats again for missing and also see if
# particular dates have tons of missing.
df_sleep['hour'] = df_sleep['date_time'].dt.hour

# could at this point create date_sleep and sleep session
# and then cut off starting at 8. or maybe don't need to 
# cut off at 8? 

np.sort(df_sleep['hour'].unique())
sleep_hours_list = [18,19,20,21,22,23,24,0,1,2,3,4,5,6,7,8,9,10,11]
df_sleep_8_to_11 = df_sleep[df_sleep['hour'].isin(sleep_hours_list)]
len(df_sleep_8_to_11) / len(df_sleep)  # 99.2% of the file remains
np.sort(df_sleep_8_to_11['hour'].unique())


# give new date that relates to the day the sleep started. 
# e.g., if tue is july 1, then that night is sleep associated with july 1

df_sleep_8_to_11.groupby('date_sleep').size().hist(alpha=.6, bins=20)
plt.grid(False)
# delete dates of sleep where fewer than 400 rows (= 3.33 hours)
df_sleep_data_per_day = df_sleep_8_to_11.groupby('date_sleep').size()
df_sleep_data_per_day = df_sleep_data_per_day.sort_values()
df_sleep_data_per_day = df_sleep_data_per_day[df_sleep_data_per_day<400]
dates_with_little_data_list = list(df_sleep_data_per_day.index)
len(df_sleep_8_to_11['date_sleep'].unique())
df_sleep_8_to_11 = df_sleep_8_to_11[-df_sleep_8_to_11['date_sleep'].isin(dates_with_little_data_list)]
len(df_sleep_8_to_11['date_sleep'].unique())

len(df_sleep_8_to_11[df_sleep_8_to_11['hr'].isnull()]) / len(df_sleep_8_to_11)
# still over 52%
for date in df_sleep_8_to_11['date_sleep'].unique():
    df_date = df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']==date]
    print(date, np.round(len(df_date[df_date['hr'].isnull()]) / len(df_date), 3))

df_sleep_8_to_11[['date_time', 'date_sleep', 'hr']][df_sleep_8_to_11['date_sleep']=='2018-09-23'].tail(50)

df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']=='2018-09-23'].tail(50)
# what happens w these nulls?
df_hr[(df_hr['date_time']>'2018-09-24 07:01:30') & (df_hr['date_time']<'2018-09-24 07:20:30')]
# there's just nothing in the hr files for these times. just missing. ok.

# PLOT RELATIONSHIP BETWEEN NUMBER OF DATAPOINTS ASLEEP IN A NIGHT AND THE MEAN HR
# TO SEE THE RANGE INCREASE AS FEWER DATA POINTS. FUNNEL PLOT

# next steps - look at notes on github jupyter nb
# clean data - remove outliers and implausible values
# put date_time as index and resample so in real time
# including gaps? but will that give all date_times
# for when i was awake during the day? what do i want here?
# for computing things like mean hr and times awake, etc.
# i don't need a real time series. it's ok if there are gaps
# but then to viz the time series and aggregated times series
# can i plot a ragged ts in relplot? maybe. play with. might
# not need to resample but will prob need to put date_time as the index

# also, first come up with a plan for what i want to present
# draw it out, on paper, with the grahps i want.

df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']=='2018-08-01']
df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']=='2018-08-02']
df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']=='2018-08-03']
df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']=='2018-08-04']
# these match up with the online dashboard

# turn awake times hr into nan
df_sleep_8_to_11['awake'].value_counts(normalize=True)
df_sleep_8_to_11[df_sleep_8_to_11['awake']==1]['hr'].mean()
df_sleep_8_to_11.loc[df_sleep_8_to_11['awake']==1, 'hr'] = np.nan

len(df_sleep_8_to_11['date_sleep'].unique())  # 699

# how to reample w 30 sec. do i have to do this? this might be unnecessary
# check to see if anything is different when resample within date
# looks like i don't need to resample -- that the sleep sessions all
# have complete set of rows 30 sec apart. can see here that the only
# days with a diff number of rows when resampled are those with multiple
# sleep sessions because the resampling fills in gaps between sleep sessions
for date in df_sleep_8_to_11['date_sleep'].unique()[110:150]:
    print(date)
    df_date = df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']==date]
    print(df_date['end_sleep'].sum())
    print(len(df_date))
    print(len(df_date.resample('30S', on='date_time').mean()))
    print()

# interpolate w limit of 1 within date_sleep, which fills in gaps where just one gap 
# actually, should be interpolating within sleep session, not date_sleep
# i.e., if 2+ sleep sessions per one date_sleep, then interpolate within each session
# (do this before resampling. because i'll groupby date_sleep and sleep session, 
# I don't want to count time between sleep sessions as a sleep session)
#df_sleep_8_to_11.head()
#df_sleep_8_to_11['sleep_session'] = df_sleep_8_to_11.groupby('date_sleep')['start_sleep'].transform(lambda x: x.expanding().sum())
#df_sleep_8_to_11[df_sleep_8_to_11['sleep_session'].isnull()]
#df_sleep_8_to_11[df_sleep_8_to_11['sleep_session'].isnull()]['date_sleep'].unique()

date = '2017-02-17'
df_date = df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']==date]
df_date = df_date.reset_index(drop=True)
df_date[df_date['start_sleep']==1]
df_date[(df_date.index>140) & (df_date.index<150)]
df_date[(df_date['date_time']>'2017-02-17 20:06:00') & (df_date['date_time']<'2017-02-17 20:11:00')]
df_date[(df_date['date_time']>'2017-02-17 22:50:30') & (df_date['date_time']<'2017-02-17 23:05:30')]

#df_sleep_8_to_11[df_sleep_8_to_11['sleep_session'].isnull()]['date_sleep'].value_counts()
# why don't i have sleep sessions for all?
for date in df_sleep_8_to_11['date_sleep'].unique():
    #print(date)
    df_date = df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']==date]
    df_date = df_date.reset_index()
    first_row_start_sleep = df_date['start_sleep'][:1].values[0]
    #sum_start_sleeps = df_date['start_sleep'].sum()
    if first_row_start_sleep == 1:
        None
    else:
        print(date)

# no start_sleep: 2017-11-26T00:00:00.000000000
date = '2018-08-26'   # '2017-10-29'  #'2017-01-28'
df_date = df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']==date]
df_date = df_date.reset_index(drop=True)
df_date[df_date['start_sleep']==1]
# start date doesn't come until index 174. why is that?
# and all of these seem to start at 8pm and don't have any
# flags for types of sleep until the start sleep point
# df_date[115:130]
# ah, is it possible that these are days i was sleeping
# before my 8pm artificial start. and so the fitbit start time
# was before 8pm. and so i don't have that here
# i might want to flag these days and omit from anys?

# for now, if no sleep_start = 1 for first row of a date_sleep, give it 1
first_index_each_date_sleep = df_sleep_8_to_11.groupby('date_sleep').apply(lambda x: x.index[0]).reset_index().rename(columns={0:'index'})
first_index_each_date_sleep['start_sleep'] = 1
index_to_start_sleep_flag_dict = dict(zip(first_index_each_date_sleep['index'], first_index_each_date_sleep['start_sleep']))
df_sleep_8_to_11['index_copy'] = df_sleep_8_to_11.index
df_sleep_8_to_11['start_of_date_sleep'] = df_sleep_8_to_11['index_copy'].map(index_to_start_sleep_flag_dict)
df_sleep_8_to_11.loc[(df_sleep_8_to_11['start_of_date_sleep']==1) & (df_sleep_8_to_11['start_sleep'].isnull())] 

df_sleep_8_to_11['start_sleep_occurs_before'] = 0
df_sleep_8_to_11.loc[(df_sleep_8_to_11['start_of_date_sleep']==1) & (df_sleep_8_to_11['start_sleep'].isnull()), 'start_sleep_occurs_before'] = 1 

# how many don't have a start_sleep flag at first row
df_sleep_8_to_11.loc[df_sleep_8_to_11['start_of_date_sleep']==1, 'start_sleep'] = 1
# now compute sleep_session again
df_sleep_8_to_11['sleep_session'] = df_sleep_8_to_11.groupby('date_sleep')['start_sleep'].transform(lambda x: x.expanding().sum())
df_sleep_8_to_11[df_sleep_8_to_11['sleep_session'].isnull()]
# great

# set hr_interpolate to hr so can use code below
#df_sleep_8_to_11['hr_interpolate'] = df_sleep_8_to_11['hr']
# ==========================
# hold off on interpoliation until later, after take out outliers
#df_sleep_8_to_11['hr_interpolate'] = df_sleep_8_to_11.groupby(['date_sleep', 'sleep_session'])['hr'].transform(lambda x: x.interpolate(limit=1))
#df_sleep_8_to_11.columns
#df_sleep_8_to_11[['date_sleep', 'date_time', 'hr', 'hr_interpolate']]
# ==========================


# to compute missing hr data and avg sleep length
df_sleep_min = df_sleep_8_to_11[df_sleep_8_to_11['date_time'].astype(str).str[-2:-1]=='0']
df_sleep_min = df_sleep_min[df_sleep_min['awake']==0]
df_sleep_min[df_sleep_min['awake'].isnull()]
len(df_sleep_min)  # 215818
1 - (len(df_sleep_min[df_sleep_min['hr'].isnull()]) / len(df_sleep_min))  # 0.96

df_hr_missing_by_day = df_sleep_min.groupby('date_sleep')['hr'].apply(lambda x: len(x[x.isnull()])) / df_sleep_min.groupby('date_sleep')['hr'].size() * 100
# take out bad day here -- almost all missing
df_hr_missing_by_day.max()
df_hr_missing_by_day[df_hr_missing_by_day==1]
df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']=='2017-09-01']

df_hr_missing_by_day = df_hr_missing_by_day[df_hr_missing_by_day<70]

df_hr_missing_by_day = df_hr_missing_by_day.round(1)

df_hr_missing_by_day.hist(bins=12, color='dodgerblue', alpha=.5)
plt.xlabel('Percent of Sleep Minutes With Missing HR', fontsize=20)
plt.ylabel('Number of Nights', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(df_hr_missing_by_day.min(), df_hr_missing_by_day.max())
plt.axvline(df_hr_missing_by_day.mean(), linestyle='--', color='grey')
plt.grid(False)
sns.despine()

# RE DO THIS WHEN USING DATA JUST FROM DAYS I HAVE ALC READINGS
# NOT SURE WHY I DON'T HAVE DATA HERE? EXPLORE.

# I SHOULD PROB LOOK AT RESTLESS OVER TIME. POSSIBLE THAT THIS NOW GETS
# CODED AS AWAKE VS IN EARLY DATA AS RESTLESS?
plt.plot(df_sleep_min.groupby('date_sleep')['restless'].mean())

df_for_sleep_stats = df_sleep_min.groupby('date_sleep').size()
df_for_sleep_stats.mean() / 60
df_for_sleep_stats = df_for_sleep_stats/60
df_for_sleep_stats.min()
df_for_sleep_stats.max()
df_for_sleep_stats.std()


# for present
#df_for_sleep_stats.hist(alpha=.5, color='dodgerblue', bins=20)
cat_list = list(range(3,16))
plt.hist(df_for_sleep_stats.values, alpha=.5, color='dodgerblue', bins=12, align='left')
plt.xticks(cat_list, cat_list, fontsize=15) 
plt.xlabel('Hours Asleep', fontsize=20)
plt.ylabel('Number of Days', fontsize=20)
plt.yticks(fontsize=15)
plt.xlim(df_for_sleep_stats.min()-.6, df_for_sleep_stats.max()+.6)
plt.axvline(df_for_sleep_stats.mean(), linestyle='--', color='grey')
plt.grid(False)
sns.despine()



# over time?
df_for_sleep_stats.head()
df_for_sleep_stats = df_for_sleep_stats.reset_index()
df_for_sleep_stats.rename(columns={0:'sleep_hours'}, inplace=True)
sns.relplot(x='date_sleep', y='sleep_hours', 
            data=df_for_sleep_stats, kind='line')

plt.plot(df_for_sleep_stats['date_sleep'], df_for_sleep_stats['sleep_hours'].rolling(window=30).mean())
plt.xticks(rotation=30)

# compute rolling mean here or after resampling? pretty sure that resampling
# won't change the number of rows per session (might be code below or above
# that checks that assumption). but go w this assumption for now.
df_sleep_8_to_11 = df_sleep_8_to_11.sort_values(by=['date_time'])
df_sleep_8_to_11['hr_rolling_30_min'] = df_sleep_8_to_11.groupby(['date_sleep', 
                'sleep_session'])['hr'].transform(lambda x: x.rolling(window=30, min_periods=3, center=True).mean())
df_sleep_8_to_11.tail(40)
df_sleep_8_to_11.head(40)

# with min_periods set to 5, this will give lots of rows data where there is
# not in actuality. so after creating this rolling avg, delete time points 
# where don't have any actual interpolated hr data

# for present
df_sleep_8_to_11['hr'].hist(alpha=.5, color='dodgerblue', bins=25, normed=True)
plt.xlabel('HR', fontsize=20)
plt.ylabel('Proportinon of\nHR measurements', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(df_sleep_8_to_11['hr'].min(), df_sleep_8_to_11['hr'].max())
#plt.axvline(df_sleep_8_to_11['hr'].mean(), linestyle='--', color='grey')
plt.grid(False)
sns.despine()


# ==========================
# do this later after calc interpolate
#df_sleep_8_to_11.loc[df_sleep_8_to_11['hr_interpolate'].isnull(), 'hr_rolling_30_min'] = np.nan
# ==========================

# doing rolling mean on df ragged ts doesn't work for me here becuase
# can't center it on a time window. 

# when does fitbit consider it a new sleep session?
df_sleep_8_to_11 = df_sleep_8_to_11.sort_values(by='date_time')
df_sleep_8_to_11['date_time_prior'] = df_sleep_8_to_11.groupby('date_sleep')['date_time'].transform(lambda x: x.shift(1))
df_sleep_8_to_11['time_since_prior_row'] = (df_sleep_8_to_11['date_time'] - df_sleep_8_to_11['date_time_prior']) / np.timedelta64(1, 'm') 
df_sleep_8_to_11[df_sleep_8_to_11['start_sleep']==1]['time_since_prior_row'].hist(alpha=.5)
plt.grid(False)

df_sleep_8_to_11[df_sleep_8_to_11['start_sleep']==1]['time_since_prior_row'].min()
df_sleep_8_to_11[df_sleep_8_to_11['start_sleep']==1]['time_since_prior_row'].max()
# looks like anything >= 1hr is considered another sleep session by fitbit

time_since_prior_sleep_list = df_sleep_8_to_11[df_sleep_8_to_11['start_sleep']==1]['time_since_prior_row'].values
np.sort(time_since_prior_sleep_list)

df_sleep_sessions = df_sleep_8_to_11.groupby('date_sleep')['sleep_session'].mean()
len(df_sleep_sessions[df_sleep_sessions>1])
# 29 (41 w total days) days have 2+ sleep sessions

# could derive my own sleep sessions from a break in time series of more than x
# so not relying on fitbit's definition of sleep session. but hold of on that for the moment
# but would be good to do that or else test fitbit's assumptions about what constitutes
# a sleep session

# on nights when have more than one sleep episode, resampling will
# increase the number of rows because it fills in the gaps between
# sleep episodes. do i want that? just experimented. if i resample
# then it plots it correctly -- with missing line for those times
# in which i wasn't sleeping. If i don't resample then it plots
# a straight line through those missing times in which i was awake
# so, sure, resample within date_sleep. 
date = '2017-02-17'
date = '2017-11-09'
df_date = df_sleep_8_to_11[df_sleep_8_to_11['date_sleep']==date]
len(df_date)
len(df_date.resample('30S', on='date_time').mean())
df_date['sleep_session'].unique()


## ========================
## resample later
#len(df_sleep_8_to_11.groupby('date_sleep').resample('30S', on='date_time').mean())  # 727693
#len(df_sleep_8_to_11)  # 713593
#df_sleep_8_to_11_resampled = df_sleep_8_to_11.groupby('date_sleep').resample('30S', on='date_time').mean().reset_index()
## this resampled within date_sleep
#df_sleep_8_to_11_resampled.head()
#len(df_sleep_8_to_11_resampled)
#df_date = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['date_sleep']==date]
#len(df_date)  # all works
## =========================

# plot raw against smooth and take smoothed when outliers
# get diff between smoothed and actual hr
df_sleep_8_to_11['hr_vs_smoothed_diff'] = df_sleep_8_to_11['hr'] - df_sleep_8_to_11['hr_rolling_30_min']

df_sleep_8_to_11['hr_vs_smoothed_diff'].hist(alpha=.5, bins=50)
plt.grid(False)

df_sleep_8_to_11[df_sleep_8_to_11['hr_vs_smoothed_diff']>0]['hr_vs_smoothed_diff'].hist(alpha=.5, bins=50)
pos_diff_mean = df_sleep_8_to_11[df_sleep_8_to_11['hr_vs_smoothed_diff']>0]['hr_vs_smoothed_diff'].mean() 
pos_diff_sd = df_sleep_8_to_11[df_sleep_8_to_11['hr_vs_smoothed_diff']>0]['hr_vs_smoothed_diff'].std()
x_sd_above_mean = pos_diff_mean + pos_diff_sd*3

df_sleep_8_to_11[df_sleep_8_to_11['hr_vs_smoothed_diff']<0]['hr_vs_smoothed_diff'].hist(alpha=.5, bins=50)
neg_diff_mean = df_sleep_8_to_11[df_sleep_8_to_11['hr_vs_smoothed_diff']<0]['hr_vs_smoothed_diff'].mean() 
neg_diff_sd = df_sleep_8_to_11[df_sleep_8_to_11['hr_vs_smoothed_diff']<0]['hr_vs_smoothed_diff'].std()
x_sd_below_mean = neg_diff_mean - neg_diff_sd*3

len(df_sleep_8_to_11['date_sleep'].unique())

# replace outliers with smoothed value
df_sleep_8_to_11['outlier_hr'] = 0
df_sleep_8_to_11.loc[(df_sleep_8_to_11['hr_vs_smoothed_diff'] > x_sd_above_mean) |
        (df_sleep_8_to_11['hr_vs_smoothed_diff'] < x_sd_below_mean) , 
        'outlier_hr'] = 1  
df_sleep_8_to_11['outlier_hr'].value_counts(normalize=True)  # .7%
df_sleep_8_to_11['hr_clean'] = df_sleep_8_to_11['hr']
df_sleep_8_to_11.loc[df_sleep_8_to_11['outlier_hr']==1, 'hr_clean'] = df_sleep_8_to_11['hr_rolling_30_min']
df_sleep_8_to_11[['hr', 'hr_clean']]
#df_sleep_8_to_11_resampled['hr_clean'] = df_sleep_8_to_11_resampled['hr']
#df_sleep_8_to_11_resampled.loc[df_sleep_8_to_11_resampled['outlier_hr']==1, 'hr_clean'] = df_sleep_8_to_11_resampled['hr_rolling_30_min']

# do iterpolation here
# this first line, when change code to hr above, just interpoliate hr here
df_sleep_8_to_11['hr_interpolate'] = df_sleep_8_to_11.groupby(['date_sleep', 'sleep_session'])['hr'].transform(lambda x: x.interpolate(limit=1))
df_sleep_8_to_11['hr_interpolate_clean'] = df_sleep_8_to_11.groupby(['date_sleep', 'sleep_session'])['hr_clean'].transform(lambda x: x.interpolate(limit=1))
# think having troupbe because this is resampled

df_sleep_8_to_11[['date_sleep', 'date_time', 'hr', 'hr_interpolate_clean']]
df_sleep_8_to_11.loc[df_sleep_8_to_11['hr_interpolate'].isnull(), 'hr_rolling_30_min'] = np.nan

# resample here?
df_sleep_8_to_11_resampled = df_sleep_8_to_11.groupby('date_sleep').resample('30S', on='date_time').mean().reset_index()
# this resampled within date_sleep
df_sleep_8_to_11_resampled.head()
len(df_sleep_8_to_11_resampled)
df_date = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['date_sleep']==date]
len(df_date)  # all works

df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['hr']==36]


# plot rolling vs raw on a day
hr_variable = 'hr_interpolate'  # 'hr_interpolate_clean'
hr_variable = 'hr_interpolate_clean'  # 'hr_interpolate'
dates_list = df_sleep_8_to_11_resampled['date_sleep'].unique()
date = dates_list[325]
#date = '2017-5-26'
#date = '2018-09-04'
#date = '2018-07-18'
#date = '2017-05-07'  # can see the low outlier
date = '2018-07-15'  # can see the low outlier
df_day = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['date_sleep']==date]
plt.plot(df_day['date_time'], df_day[hr_variable], 
         alpha=.4, color='green', linewidth=1)
plt.plot(df_day['date_time'], df_day['hr_rolling_30_min'], 
         alpha=.5, color='grey', linewidth=3)  
plt.grid(axis='y', alpha=.4)
plt.ylim(df_day[hr_variable].min()-1,df_day[hr_variable].max()+1)


# for present
# plot the hr raw alone
df_day = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['date_sleep']==date]
ax = plt.figure(figsize=(12,6), dpi=80)  # figsize=(13, 6)
plt.plot(df_day['date_time'], df_day['hr_interpolate'], 
         alpha=.5, color='green', linewidth=1.5)
plt.grid(axis='y', alpha=.5)
plt.ylim(35,75)
plt.ylabel('Heart Rate', fontsize=26)
plt.xlabel('Time', fontsize=26)
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))  
ax.autofmt_xdate()
sns.despine()

# for present w smoothed line
ax = plt.figure(figsize=(12,6), dpi=80)  # figsize=(13, 6)
plt.plot(df_day['date_time'], df_day['hr_interpolate'], 
         alpha=.4, color='green', linewidth=1.25)
plt.plot(df_day['date_time'], df_day['hr_rolling_30_min'], 
         alpha=.5, color='grey', linewidth=4)  
plt.grid(axis='y', alpha=.5)
plt.ylim(35,75)
plt.ylabel('Heart Rate', fontsize=26)
plt.xlabel('Time', fontsize=26)
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))  
ax.autofmt_xdate()
sns.despine()

# for present w smoothed line and cleaned
ax = plt.figure(figsize=(12,6), dpi=80)  # figsize=(13, 6)
plt.plot(df_day['date_time'], df_day['hr_interpolate_clean'], 
         alpha=.4, color='green', linewidth=1.55)
#plt.plot(df_day['date_time'], df_day['hr_rolling_30_min'], 
#         alpha=.5, color='grey', linewidth=4)  
plt.grid(axis='y', alpha=.5)
plt.ylim(35,75)
plt.ylabel('Heart Rate', fontsize=26)
plt.xlabel('Time', fontsize=26)
plt.yticks(fontsize=22)
plt.xticks(fontsize=22)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))  
ax.autofmt_xdate()
sns.despine()


hr_threshold = 40  # 50
print(len(df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['hr_interpolate']<hr_threshold]['date_sleep'].unique()))
print(len(df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['hr_interpolate_clean']<hr_threshold]['date_sleep'].unique()))


# plot all nights, aggregated
df_sleep_8_to_11_resampled.head()
# For aggregating: give all one bogus date for pm and subsequent bogus date for am
df_sleep_8_to_11_resampled['hour'] = df_sleep_8_to_11_resampled['date_time'].dt.hour
df_sleep_8_to_11_resampled['date_bogus'] = np.nan
df_sleep_8_to_11_resampled.loc[(df_sleep_8_to_11_resampled['hour']<=23) & 
                               (df_sleep_8_to_11_resampled['hour']>=18), 'date_bogus'] = '2018-10-01'
df_sleep_8_to_11_resampled.loc[(df_sleep_8_to_11_resampled['hour']>=0) & 
                               (df_sleep_8_to_11_resampled['hour']<=11), 'date_bogus'] = '2018-10-02'
df_sleep_8_to_11_resampled.head()
df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['date_bogus'].isnull()]['hour']
df_sleep_8_to_11_resampled['date_bogus'].unique()
df_sleep_8_to_11_resampled['date_time_bogus'] = pd.to_datetime(df_sleep_8_to_11_resampled['date_bogus']+' '+df_sleep_8_to_11_resampled['date_time'].dt.time.astype(str))

g = sns.relplot(x='date_time_bogus', y='hr_interpolate_clean', data=df_sleep_8_to_11_resampled, kind='line', ci=None)
g.fig.autofmt_xdate()


# plot aggregated for present
#fig, ax = plt.figure(figsize=(12,6), dpi=80)  # figsize=(13, 6)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=80)
sns.relplot(x='date_time_bogus', y='hr_interpolate_clean', 
            data=df_sleep_8_to_11_resampled,  # .sample(n=1000), 
            kind='line', ci='sd', ax=ax, color='green')
# assign locator and formatter for the xaxis ticks.
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#axis.xaxis.set_major_locator(HourLocator(byhour))
ax.set_xlabel('Time', fontsize=26)
ax.set_ylabel('Heart Rate', fontsize=26)
#ax.set_yticklabels(list(range(50,70,2)))
#yticks = list(range(48,66,2))
yticklabels = [str(tick) for tick in yticks]
#ax.set_yticks(yticks)
#ax.set_yticklabels(yticklabels)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.set_xlim('2018-10-01 21:00:00', '2018-10-02 10:00:00')
ax.set_ylim(46,69)
ax.grid(axis='y', alpha=.5)
fig.autofmt_xdate()
sns.despine()


# for present
len(df_sleep_8_to_11_resampled['date_sleep'].unique())
# overlay smoothed for many nights
alpha_level = .075
width_for_line = 1
yticks = list(range(40,80,5))
yticklabels = [str(tick) for tick in yticks]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(11,6), dpi=80)
for date in df_sleep_8_to_11_resampled['date_sleep'].unique()[:]:
    df_date = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['date_sleep']==date]
    plt.plot(df_date['date_time_bogus'], df_date['hr_rolling_30_min'], 
             alpha=alpha_level, color='green', linewidth=width_for_line)
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
fig.autofmt_xdate()
ax.set_xlabel('Time', fontsize=26)
ax.set_ylabel('Heart Rate', fontsize=26)
ax.set_xlim('2018-10-01 22:00:00', '2018-10-02 11:00:00')
ax.set_ylim(43,72)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
sns.despine()
# use this to grab a few outliers -- what's going on here?


# plot select nights in grid
#dates_to_plot_list = list(df_sleep_8_to_11_resampled['date_sleep'].unique())
#dates_to_plot_list = dates_to_plot_list[100:105]
#sns.relplot(x='date_time_bogus', y='hr_interpolate_clean',
#            col='date_sleep', col_wrap=1,
#            height=3, aspect=4, linewidth=2.5, kind='line', 
#            data=df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['date_sleep'].isin(dates_to_plot_list)]);
#
            
            
            
# ------------
# do some aggregate anys:
# subjective sleep ratings
# alcohol
# fun ratings
# energy ratings (or just fun, if this reflects poorly on me?)

# then viz in ts with ci


# row = night's sleep
df_hr_mean = df_sleep_8_to_11_resampled.groupby('date_sleep')['hr_clean'].mean().reset_index().rename(columns={'hr_clean':'hr_mean'})
df_hr_mean.head()

df_hr_mean['hr_mean'].hist(alpha=.5, bins=25, color='dodgerblue' )
plt.grid(False)
plt.axvline(df_hr_mean['hr_mean'].mean(), linestyle='--', color='black', linewidth=1, alpha=.75)


# ----------------------------------------------------------------------------
# imported daily ratings earlier already

# import daily ratings
# get daily questions -
#df_daily_qs_early = pd.read_excel('Mood Measure (Responses).xlsx')
#df_daily_qs_early.head()
#
#df_daily_qs_early = df_daily_qs_early[df_daily_qs_early['What are your initials']=='AM']
#df_daily_qs_early['date'] = pd.to_datetime(df_daily_qs_early['Timestamp'].dt.year.astype(str) + '-' + df_daily_qs_early['Timestamp'].dt.month.astype(str) + '-' + df_daily_qs_early['Timestamp'].dt.day.astype(str))
#df_daily_qs_early['date_short'] = df_daily_qs_early['date'].dt.date
#df_daily_qs_early.head()
#
## qs to use and merge with current qs?
#df_daily_qs_early.dtypes
#df_daily_qs_early = df_daily_qs_early[['date', 'Today, are you tired?', 'Last night, did you sleep well?']]
#df_daily_qs_early['energy'] = 5 - df_daily_qs_early['Today, are you tired?']
#df_daily_qs_early.rename(columns={'Last night, did you sleep well?':'sleep'}, inplace=True)
#df_daily_qs_early = df_daily_qs_early[['date', 'energy', 'sleep']]
#
#
#df_daily_qs_current = pd.read_excel('Daily_Measure_2018_10_8.xlsx')
##df_daily_qs_current = pd.read_csv('Daily_Measure_from_2_26_17.csv')
#df_daily_qs_current.head()
#df_daily_qs_current.tail()
#
#df_daily_qs_current['Timestamp'] = pd.to_datetime(df_daily_qs_current['Timestamp'])
#df_daily_qs_current['year'] = df_daily_qs_current['Timestamp'].dt.year.astype(str).str.split('.').str[0]
#df_daily_qs_current['month'] = df_daily_qs_current['Timestamp'].dt.month.astype(str).str.split('.').str[0]
#df_daily_qs_current['day'] = df_daily_qs_current['Timestamp'].dt.day.astype(str).str.split('.').str[0]
#df_daily_qs_current['date'] = df_daily_qs_current['year'] + '-' + df_daily_qs_current['month'] + '-' + df_daily_qs_current['day']
#df_daily_qs_current['date'].replace('nan-nan-nan', np.nan, inplace=True)
#df_daily_qs_current['date'] = pd.to_datetime(df_daily_qs_current['date'])
#df_daily_qs_current.columns
#
#df_daily_qs_current = df_daily_qs_current[['date', 'Right now I feel energetic.',
#       'At some point today I felt annoyed.', 'At some point today I had fun with someone.',
#       'Last night, I woke up from a restorative sleep.', 'Last night I had ____ alcoholic drinks',
#       'The alcohol I drank last night was ____', 'Notes', 'Right now I feel tired.', 
#       'At some point today I felt insulted.']]
#
#df_daily_qs_current.columns
#df_daily_qs_current.columns = ['date', 'energy', 'annoyed', 'fun', 'sleep', 'alcohol', 'alcohol_type', 'notes', 'tired', 'insulted']
#
#for col in df_daily_qs_current.columns:
#    print(col, len(df_daily_qs_current[df_daily_qs_current[col].isnull()]))
#
#for col in df_daily_qs_current.columns:
#    print(col, len(df_daily_qs_current[df_daily_qs_current[col].notnull()]))
## given that i have annoyed and energy for just about all rows, i can delete tired and insulted
#df_daily_qs_current = df_daily_qs_current[['date', 'energy', 'annoyed', 'fun', 
#                                           'sleep', 'alcohol', 'alcohol_type',
#                                           'notes']]
#
#df_daily_qs_current['alcohol']
#
#df_daily_qs = pd.concat([df_daily_qs_early, df_daily_qs_current], ignore_index=True)
#df_daily_qs.head(20)
#
## 6 duplicate dates
#len(df_daily_qs['date'].unique())
#len(df_daily_qs)
#
#dates_duplicated_list = df_daily_qs[df_daily_qs['date'].duplicated()]['date'].values
#df_daily_qs[df_daily_qs['date'].isin(dates_duplicated_list)]
#df_daily_qs = df_daily_qs.drop_duplicates(subset='date', keep='last')
#df_daily_qs.tail()
#
## the alcohol from the date on df_daily_qs refers to previous date
#df_daily_qs['date_prior'] = df_daily_qs['date'] - timedelta(days=1)
#df_daily_qs['date_two_prior'] = df_daily_qs['date'] - timedelta(days=2)
#
#df_daily_qs[['date_prior', 'date_two_prior', 'date']]
#date_to_alcohol_dict = dict(zip(df_daily_qs['date_prior'], df_daily_qs['alcohol']))
# ----------------------------------------------------------------------------


df_hr_mean['alcohol'] = np.nan
df_hr_mean['alcohol'] = df_hr_mean['date_sleep'].map(date_to_alcohol_dict)
len(df_hr_mean[df_hr_mean['alcohol'].notnull()])

# for present
df_pct_alcohol = df_hr_mean['alcohol'].value_counts(normalize=True).reset_index()
sns.barplot(x='index', y='alcohol', data=df_pct_alcohol, alpha=.6, 
            color='dodgerblue')
plt.ylabel('Proportion of Days', fontsize=20)
plt.xlabel('Number of Drinks', fontsize=20)
plt.xticks([0,1,2,3,4,5], ['0','1','2','3','4','5'], fontsize=16)
plt.yticks(fontsize=16)
sns.despine()


df_hr_mean['alcohol'].replace([5,6],[4,4], inplace=True)

sns.countplot(x='alcohol', data=df_hr_mean, alpha=.7, color='dodgerblue')
plt.xlabel('Number of Drinks', fontsize=15)
plt.ylabel('Number of Days', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()


df_pct_alcohol = df_hr_mean['alcohol'].value_counts(normalize=True).reset_index()
# for present
sns.barplot(x='index', y='alcohol', data=df_pct_alcohol, alpha=.6, 
            color='dodgerblue')
plt.ylabel('Proportion of Days', fontsize=15)
plt.xlabel('Number of Drinks', fontsize=15)
plt.xticks([0,1,2,3,4], ['0','1','2','3','4+'], fontsize=14)
plt.yticks(fontsize=14)
sns.despine()


df_hr_mean['alcohol_tertile'] = df_hr_mean['alcohol']
df_hr_mean['alcohol_tertile'].replace([3,4,5,6],[2,2,2,2], inplace=True)
df_hr_mean['alcohol_tertile'].value_counts()
df_hr_mean['alcohol_tertile'] = df_hr_mean['alcohol_tertile'].astype(str)
df_hr_mean['alcohol_tertile'] = df_hr_mean['alcohol_tertile'].str.split('.').str[0]
df_hr_mean['alcohol_tertile'].replace('nan', np.nan, inplace=True)
df_pct_alcohol = df_hr_mean['alcohol_tertile'].value_counts(normalize=True).reset_index()

sns.barplot(x='index', y='alcohol_tertile', data=df_pct_alcohol, alpha=.6, 
            color='dodgerblue')
plt.ylabel('Proportion of Days', fontsize=15)
plt.xlabel('Number of Drinks', fontsize=15)
plt.yticks(fontsize=14)
plt.xticks([0,1,2], ['0','1','2+'], fontsize=14)
sns.despine()


# CONVER THIS TO 0, 1, 2+ PRESENT GRAPH?
sns.barplot(x='alcohol', y='hr_mean', data=df_hr_mean, alpha=.6, 
            color='dodgerblue', errcolor='grey')
plt.ylim(40,65)
plt.ylabel('Mean Heart Rate', fontsize=15)
plt.xlabel('Number of Drinks', fontsize=15)
plt.xticks([0,1,2,3,4], ['0','1','2','3','4+'], fontsize=14)
plt.yticks(fontsize=14)
sns.despine()

df_hr_mean['alcohol_tertile'].value_counts()
sns.barplot(x='alcohol_tertile', y='hr_mean', data=df_hr_mean, alpha=.6, 
            color='dodgerblue', errcolor='grey')  # order=['no_drinks', 'one_drink', 'two_plus_drinks']
plt.ylim(40,65)
plt.ylabel('Mean Heart Rate', fontsize=15)
plt.xlabel('Number of Drinks', fontsize=15)
#plt.xticks([0,1,2], ['0','1','2','3','4+'], fontsize=14)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()

df_hr_mean.groupby('alcohol_tertile')['hr_mean'].std()

results = smf.ols(formula = 'hr_mean ~ alcohol', data=df_hr_mean).fit()
print(results.summary())
dir(results)
results.pvalues.alcohol

df_hr_mean[['hr_mean', 'alcohol']].corr()
# wow, .54 corr! cohen says this is 'big'

# effect 
df_hr_mean['alcohol_dichot'] = np.nan
df_hr_mean.loc[df_hr_mean['alcohol']>0, 'alcohol_dichot'] = 1
df_hr_mean.loc[df_hr_mean['alcohol']==0, 'alcohol_dichot'] = 0

n_total = len(df_hr_mean[df_hr_mean['alcohol_dichot'].notnull()])
n_control = len(df_hr_mean[df_hr_mean['alcohol_dichot']==0])
n_treatment = len(df_hr_mean[df_hr_mean['alcohol_dichot']==1])
results = smf.ols(formula = 'hr_mean ~ alcohol_dichot', data=df_hr_mean).fit()
t_value = results.tvalues[1]
cohens_d = t_value * np.sqrt( ((n_treatment + n_control)/(n_treatment * n_control)) * ((n_treatment + n_control) / (n_treatment + n_control - 2) ))
# large to very large

# control for day of the week?
# could get matched comparison group of day of the week. but not 1:1. 1:many

# first vs. second half of the night?

# control for confounds -- what's something that could vary with alcohold
# and affect hr while sleeping?
# day of week, season, temp, time of sleep onset, length of sleep, trend over time
# how to think about lag variables? hr from prior day(s) could corr with alcohol
# but also affect hr. 
df_hr_mean.tail()
df_hr_mean['day_of_week'] = df_hr_mean['date_sleep'].dt.dayofweek
df_hr_mean['day_name'] = df_hr_mean['day_of_week'].replace([0,1,2,3,4,5,6], 
          ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])

sns.barplot(x='day_name', y='hr_mean', data=df_hr_mean, 
            color='dodgerblue', alpha=.6, errcolor='grey', 
            order=['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])
plt.ylim(40,60)

df_hr_mean = pd.concat([df_hr_mean, pd.get_dummies(df_hr_mean['day_name'])], axis=1)


#df_hr_mean = df_hr_mean[['date_sleep', 'hr_mean', 'alcohol', 'alcohol_tertile', 
#                         'alcohol_dichot', 'day_of_week', 'day_name']]

results = smf.ols(formula = 'hr_mean ~ alcohol', data=df_hr_mean).fit()
print(results.summary())

results = smf.ols(formula = """hr_mean ~ alcohol + tue + wed + thu + fri + 
                  sat + sun""", data=df_hr_mean).fit()
print(results.summary())
# effect for fri and sat, but for alcohol too

# look at hr over the last two years. seasonal? general trend over time?
df_hr_mean = df_hr_mean.sort_values(by='date_sleep')
df_hr_mean['hr_smoothed'] = df_hr_mean['hr_mean'].rolling(window=30, min_periods=3, center=True).mean()
df_hr_mean['hr_smoothed_2'] = df_hr_mean['hr_mean'].rolling(window=60, min_periods=3, center=True).mean()

sns.relplot(x='date_sleep', y='hr_mean', data=df_hr_mean, kind='line', alpha=.15)

g = sns.relplot(x='date_sleep', y='hr_smoothed', data=df_hr_mean, kind='line', alpha=.75)
g.fig.autofmt_xdate()
plt.ylim(50,56)

g = sns.relplot(x='date_sleep', y='hr_smoothed_2', data=df_hr_mean, kind='line', alpha=.75)
g.fig.autofmt_xdate()
plt.ylim(50,56)
# looks like seasonal trend - winter hr it's higher than summer hr
# get avg temp of month. the high.
df_hr_mean['month'] = df_hr_mean['date_sleep'].dt.month
month_to_temp_dict = {1:39, 2:42, 3:50, 4:62, 5:72, 6:80, 
                      7:85, 8:84, 9:76, 10:65, 11:54, 12:44}
df_hr_mean['temp'] = df_hr_mean['month'].map(month_to_temp_dict)

sns.barplot(x='month', y='hr_mean', data=df_hr_mean, 
            color='dodgerblue', alpha=.6, errcolor='grey')
plt.ylim(40,60)

sns.relplot(x='temp', y='hr_mean', data=df_hr_mean, kind='line')

results = smf.ols(formula = """hr_mean ~ alcohol + tue + wed + thu + fri + 
                  sat + sun + temp""", data=df_hr_mean).fit()
print(results.summary())


# what about just a general trend and also lagged variables?
# looks like maybe just a general downward trend. how to model that?
# just with index as a times point?
df_hr_mean = df_hr_mean.reset_index()
# including the index introduces pot multicolinearity. what if i did month number?
results = smf.ols(formula = """hr_mean ~ alcohol + tue + wed + thu + fri + 
                  sat + sun + temp + index""", data=df_hr_mean).fit()
print(results.summary())

df_hr_mean['year'] = df_hr_mean['date_sleep'].dt.year
df_hr_mean['month_ts'] = np.nan
#df_hr_mean.loc[df_hr_mean['year']==2016, 'month_ts'] = df_hr_mean['month']
#df_hr_mean.loc[df_hr_mean['year']==2017, 'month_ts'] = df_hr_mean['month']+12
#df_hr_mean.loc[df_hr_mean['year']==2018, 'month_ts'] = df_hr_mean['month']+24
#sns.countplot(x='month_ts', data=df_hr_mean, color='dodgerblue', alpha=.7)
#df_hr_mean['month_ts'].value_counts()
#df_hr_mean[['date_sleep', 'month_ts']]

df_month_ordered = df_hr_mean.groupby(['year', 'month']).size().reset_index()
df_month_ordered['year_month'] = df_month_ordered['year'].astype(str)+' '+df_month_ordered['month'].astype(str)
year_month_to_month_ordered_dict = dict(zip(df_month_ordered['year_month'], df_month_ordered.index))
df_hr_mean['year_month'] = df_hr_mean['year'].astype(str)+' '+df_hr_mean['month'].astype(str)
df_hr_mean['month_ts'] = df_hr_mean['year_month'].map(year_month_to_month_ordered_dict)

results = smf.ols(formula = """hr_mean ~ alcohol + tue + wed + thu + fri + 
                  sat + sun + temp + month_ts""", data=df_hr_mean).fit()
print(results.summary())
# this got rid of multicolinearity warning

sns.lmplot(x='month_ts', y='hr_mean', data=df_hr_mean, scatter_kws={'alpha':.2})
# should be a neg relationship but it's pos in the regression. ok, it's sign
# gets flipped when it's in there with all the other variables
results = smf.ols(formula = """hr_mean ~ month_ts""", data=df_hr_mean).fit()
print(results.summary())
# and it's alcohol that flips it (not any of the other variables)
results = smf.ols(formula = """hr_mean ~ month_ts + alcohol""", data=df_hr_mean).fit()
print(results.summary())
# think that's prob because i'm drinking less and less alcohol
# that's the reason there's a reduction in hr over time, and not because
# there's a real trend towads lower hr-sleep. in fact, if anything, hr-sleep
# is increasing on days that not consuming alcohol? that might be normal for aging?

# get sleep onset time. (and length of sleep.)
# get dict of date to start sleep time. could filter to first row for ea date
# 
# 'start_sleep'

df_start_sleep_time = df_sleep_8_to_11_resampled.groupby('date_sleep').head(1).reset_index()
df_start_sleep_time = df_start_sleep_time[['date_sleep', 'date_time', 'date_time_bogus']]
df_start_sleep_time['start_time'] = pd.to_datetime('2018-10-01 18:00:00')
df_start_sleep_time['start_sleep'] = (df_start_sleep_time['date_time_bogus'] - df_start_sleep_time['start_time']) / np.timedelta64(1, 'm') 
df_start_sleep_time['start_sleep'].hist()
date_to_start_sleep_dict = dict(zip(df_start_sleep_time['date_sleep'], df_start_sleep_time['start_sleep']))
df_hr_mean['start_sleep_time'] = df_hr_mean['date_sleep'].map(date_to_start_sleep_dict)

sns.relplot(x='start_sleep_time', y='hr_mean', data=df_hr_mean, kind='line')
sns.lmplot(x='start_sleep_time', y='hr_mean', data=df_hr_mean, 
           scatter_kws={'alpha':.2}, lowess=True)

sns.barplot(x='start_sleep_time', y='hr_mean', color='blue', alpha=.8, data=df_hr_mean, ci=None)
plt.ylim(40,65)


results = smf.ols(formula = """hr_mean ~ alcohol + tue + wed + thu + fri + 
                  sat + sun + temp + month_ts + start_sleep_time + 
                  I(start_sleep_time**2)""", data=df_hr_mean).fit()
print(results.summary())

df_rows_asleep = df_sleep_8_to_11[df_sleep_8_to_11['awake']!=1].groupby('date_sleep').size().reset_index().rename(columns={0:'alseep_30sec'})
df_rows_asleep['alseep_30sec'].hist(bins=20)
df_rows_asleep['alseep_30sec'].mean() / 120
df_rows_asleep['alseep_30sec'].median() / 120
df_rows_asleep['min_asleep'] = df_rows_asleep['alseep_30sec']/2
date_to_min_asleep_dict = dict(zip(df_rows_asleep['date_sleep'], df_rows_asleep['min_asleep']))
df_hr_mean['min_asleep'] = df_hr_mean['date_sleep'].map(date_to_min_asleep_dict)
df_hr_mean['min_asleep'].hist()
df_hr_mean['min_asleep'].mean()/60

sns.lmplot(x='min_asleep', y='hr_mean', data=df_hr_mean, lowess=True)
sns.lmplot(x='min_asleep', y='hr_mean', data=df_hr_mean[df_hr_mean['min_asleep']<700], lowess=True)


results = smf.ols(formula = """hr_mean ~ alcohol + tue + wed + thu + fri + 
                  sat + sun + temp + I(temp**2) + month_ts + start_sleep_time + 
                  I(start_sleep_time**2) + min_asleep + I(min_asleep**2)""", data=df_hr_mean).fit()
print(results.summary())


df_sleep_8_to_11.columns
df_sleep_8_to_11['date_time']
df_sleep_8_to_11['awake'].value_counts()
df_sleep_8_to_11['start_sleep']


df_rows_within_sleep = df_sleep_8_to_11.groupby('date_sleep').size().reset_index().rename(columns={0:'within_sleep_30sec'})
df_rows_within_sleep['within_sleep_30sec'].hist(bins=20)
df_rows_within_sleep['within_sleep_30sec'].mean() / 120
df_rows_within_sleep['within_sleep_30sec'].median() / 120
df_rows_within_sleep['min_within_sleep'] = df_rows_within_sleep['within_sleep_30sec']/2
date_to_min_within_sleep_dict = dict(zip(df_rows_within_sleep['date_sleep'], df_rows_within_sleep['min_within_sleep']))
df_hr_mean['min_within_sleep'] = df_hr_mean['date_sleep'].map(date_to_min_within_sleep_dict)
df_hr_mean['min_within_sleep'].hist()
df_hr_mean['min_within_sleep'].mean()/60

# pickle time within sleep -- from start to end of sleep including times awake in between
with open('date_to_min_within_sleep_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_min_within_sleep_dict, picklefile)


# what about including lagged hr-sleep? look up ts. is that all they'd do?
# include lagged
df_hr_mean.head()
df_hr_mean.columns
# do again so if missing days, the lag is nan

df_hr_mean = create_lagged_date_variables(3, 'date_sleep', df_hr_mean)
date_sleep_to_hr_sleep_dict = dict(zip(df_hr_mean['date_sleep'], df_hr_mean['hr_mean']))
date_sleep_to_alcohol_dict = dict(zip(df_hr_mean['date_sleep'], df_hr_mean['alcohol']))

df_hr_mean['hr_mean_lag1'] = df_hr_mean['date_lag_1'].map(date_sleep_to_hr_sleep_dict)
df_hr_mean['hr_mean_lag2'] = df_hr_mean['date_lag_2'].map(date_sleep_to_hr_sleep_dict)
df_hr_mean['hr_mean_lag3'] = df_hr_mean['date_lag_3'].map(date_sleep_to_hr_sleep_dict)
df_hr_mean['alcohol_lag1'] = df_hr_mean['date_lag_1'].map(date_sleep_to_alcohol_dict)
df_hr_mean['alcohol_lag2'] = df_hr_mean['date_lag_2'].map(date_sleep_to_alcohol_dict)
df_hr_mean['alcohol_lag3'] = df_hr_mean['date_lag_3'].map(date_sleep_to_alcohol_dict)
# below is bad way to create lagged variables because of gaps in the timeseries:
#df_hr_mean['hr_mean_lag1'] = df_hr_mean['hr_mean'].shift(1)
#df_hr_mean['hr_mean_lag2'] = df_hr_mean['hr_mean'].shift(2)
#df_hr_mean['hr_mean_lag3'] = df_hr_mean['hr_mean'].shift(3)
#df_hr_mean['alcohol_lag1'] = df_hr_mean['alcohol'].shift(1)
#df_hr_mean['alcohol_lag2'] = df_hr_mean['alcohol'].shift(2)
#df_hr_mean['alcohol_lag3'] = df_hr_mean['alcohol'].shift(3)
df_hr_mean[['hr_mean', 'hr_mean_lag1', 'hr_mean_lag2', 'hr_mean_lag3', 'alcohol', 
            'alcohol_lag1', 'alcohol_lag2']][300:310]
df_hr_mean[['hr_mean', 'hr_mean_lag1', 'hr_mean_lag2', 'hr_mean_lag3',
            'alcohol', 'alcohol_lag1', 'alcohol_lag2', 'alcohol_lag3']].corr()

#from pandas.plotting import lag_plot
#lag_plot(df_hr_mean['hr_mean'], lag=3)
#lag_plot(df_hr_mean['alcohol'], lag=3)

from pandas.plotting import autocorrelation_plot
hr_series = pd.Series(df_hr_mean['hr_mean'].values)
hr_series.ffill(inplace=True)
alcohol_series = pd.Series(df_hr_mean['alcohol'].values)
alcohol_series.ffill(inplace=True)
alcohol_series = alcohol_series[alcohol_series.notnull()]

autocorrelation_plot(hr_series)
autocorrelation_plot(alcohol_series)

autocorrelation_plot(hr_series)
plt.xlim(1,25)
plt.ylim(-.5,.5)
autocorrelation_plot(alcohol_series)
plt.xlim(1,25)
plt.ylim(-.5,.5)
plt.grid(False)

# but really i want the partial autocorrelation: The partial 
# autocorrelation at lag k is the correlation that results 
# after removing the effect of any correlations due to the 
# terms at shorter lags. 
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(hr_series, lags=20)  # 1-2 days back seem correlated.
plot_pacf(alcohol_series, lags=20)  # no days back seem relevant. use 1 lag still.


results = smf.ols(formula = """hr_mean ~ alcohol + tue + wed + thu + fri + 
                  sat + sun + temp + I(temp**2) + month_ts + start_sleep_time + 
                  I(start_sleep_time**2) + min_asleep + I(min_asleep**2) + 
                  hr_mean_lag1 + hr_mean_lag2 + alcohol_lag1
                  """, data=df_hr_mean).fit()
print(results.summary())

# can still get effect size from this t-val of alcohol?
# what about if include the subjective fun and energey and sleep ratings from day before
# and then can also see if sleep affects these things. maybe that's a separate analysis/project.

df_hr_mean.columns

# add in subjective sleep quality, fun, and energy ratings
# get date the next day. because want to see hr at night and ratings the next day
#df_daily_qs['date_next'] = df_daily_qs['date'] + timedelta(days=1)
#df_daily_qs[['date_next', 'date']]
#df_daily_qs.columns

date_to_subjective_sleep_dict = dict(zip(df_daily_qs['date_lag_1'], df_daily_qs['sleep']))
date_to_fun_dict = dict(zip(df_daily_qs['date_lag_1'], df_daily_qs['fun']))
date_to_energy_dict = dict(zip(df_daily_qs['date_lag_1'], df_daily_qs['energy']))

df_hr_mean['subj_sleep'] = df_hr_mean['date_sleep'].map(date_to_subjective_sleep_dict)
len(df_hr_mean[df_hr_mean['subj_sleep'].notnull()])

sns.countplot(x='subj_sleep', data=df_hr_mean, alpha=.7, color='dodgerblue')

sns.barplot(x='subj_sleep', y='hr_mean', data=df_hr_mean, alpha=.6, 
            color='dodgerblue', errcolor='grey')
plt.ylim(40,65)
plt.ylabel('Mean Heart Rate', fontsize=15)
plt.xlabel('Subjective Sleep Quality', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()


df_hr_mean['energy'] = df_hr_mean['date_sleep'].map(date_to_energy_dict)
len(df_hr_mean[df_hr_mean['energy'].notnull()])

sns.countplot(x='energy', data=df_hr_mean, alpha=.7, color='dodgerblue')
df_hr_mean['energy'].value_counts()
df_hr_mean['energy'].replace([0,-1], [np.nan, np.nan], inplace=True)

sns.barplot(x='energy', y='hr_mean', data=df_hr_mean, alpha=.6, 
            color='dodgerblue', errcolor='grey')
plt.ylim(40,65)
plt.ylabel('Mean Heart Rate', fontsize=15)
plt.xlabel('Energy', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()


df_hr_mean['fun'] = df_hr_mean['date_sleep'].map(date_to_fun_dict)
len(df_hr_mean[df_hr_mean['fun'].notnull()])

sns.countplot(x='fun', data=df_hr_mean, alpha=.7, color='dodgerblue')
df_hr_mean['fun'].value_counts()

sns.barplot(x='fun', y='hr_mean', data=df_hr_mean, alpha=.6, 
            color='dodgerblue', errcolor='grey')
plt.ylim(40,65)
plt.ylabel('Mean Heart Rate', fontsize=15)
plt.xlabel('Fun', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()

df_hr_mean[['energy', 'subj_sleep', 'fun', 'hr_mean', 'index']].corr()

# lag ratings 
df_hr_mean['energy_lag1'] = df_hr_mean['date_lag_1'].map(date_to_energy_dict)
df_hr_mean['energy_lag2'] = df_hr_mean['date_lag_2'].map(date_to_energy_dict)
df_hr_mean['fun_lag1'] = df_hr_mean['date_lag_1'].map(date_to_fun_dict)
df_hr_mean['fun_lag2'] = df_hr_mean['date_lag_2'].map(date_to_fun_dict)
df_hr_mean['subj_sleep_lag1'] = df_hr_mean['date_lag_1'].map(date_to_subjective_sleep_dict)
df_hr_mean['subj_sleep_lag2'] = df_hr_mean['date_lag_2'].map(date_to_subjective_sleep_dict)
# subj sleep lag1 refers to how think slept the night before the night of the dv
# so that's what i'd want to include. and perhaps a lag2 also

results = smf.ols(formula = """hr_mean ~ alcohol + tue + wed + thu + fri + 
                  sat + sun + temp + I(temp**2) + month_ts + start_sleep_time + 
                  I(start_sleep_time**2) + 
                  hr_mean_lag1 + hr_mean_lag2 + hr_mean_lag3 + alcohol_lag1 + 
                  subj_sleep_lag1 + fun_lag1""", data=df_hr_mean).fit()
print(results.summary())
# left out min_asleep + I(min_asleep**2) + 
# doesn't change things and not sure i should control for this since
# it's at the same time as the hr mean is calculated, right?


# get the model i want with all
# get alcohol 0, 1, 2+
df_hr_mean = df_hr_mean[df_hr_mean['alcohol'].notnull()]
df_hr_mean['alcohol_tertile'].replace(['0','1','2'], ['no_drinks', 'one_drink', 'two_plus_drinks'], inplace=True)
df_hr_mean['alcohol_tertile'].value_counts(normalize=True)
df_hr_mean = pd.concat([df_hr_mean, pd.get_dummies(df_hr_mean['alcohol_tertile'])], axis=1)
df_hr_mean.head()
#df_hr_mean[['no_drinks', 'one_drink']]
#del df_hr_mean['two_plus_drinks']

# for present
# model control for confounds
results = smf.ols(formula = """hr_mean ~ one_drink + two_plus_drinks + 
                  tue + wed + thu + fri + sat + sun + temp + month_ts + 
                  start_sleep_time + hr_mean_lag1 + hr_mean_lag2 + 
                  alcohol_lag1 + fun_lag1""", data=df_hr_mean).fit()
print(results.summary())

# model plain
results = smf.ols(formula = """hr_mean ~ one_drink + two_plus_drinks""", data=df_hr_mean).fit()
print(results.summary())


# get all vars set to answer any qs
# does hr from prior night affect alc next night?

date_sleep_to_start_sleep_time_dict = dict(zip(df_hr_mean['date_sleep'], df_hr_mean['start_sleep_time']))
df_hr_mean['start_sleep_time_lag'] = df_hr_mean['date_lag_1'].map(date_sleep_to_start_sleep_time_dict)

results = smf.ols(formula = """alcohol ~ hr_mean_lag1""", data=df_hr_mean).fit()
print(results.summary())

results = smf.ols(formula = """alcohol ~ hr_mean_lag1 + tue + wed + thu + fri + 
                  sat + sun + temp + I(temp**2) + start_sleep_time_lag +
                  I(start_sleep_time_lag**2) + alcohol_lag1 + hr_mean_lag2 + 
                  hr_mean_lag3""", data=df_hr_mean).fit()
print(results.summary())
# if anything, if i haven't slept well (high hr prior night) then i don't drink
# maybe go to bet early?
results = smf.ols(formula = """start_sleep_time ~ hr_mean_lag1 + tue + wed + thu + fri + 
                  sat + sun + temp + I(temp**2) + start_sleep_time_lag +
                  I(start_sleep_time_lag**2) + alcohol_lag1 + hr_mean_lag2 + 
                  hr_mean_lag3""", data=df_hr_mean).fit()
print(results.summary())
# but not statistically signif

# any Xs i ight look at?
results = smf.ols(formula = """alcohol ~ hr_mean_lag1 + tue + wed + thu + fri + 
                  sat + sun + temp + I(temp**2) + start_sleep_time_lag +
                  I(start_sleep_time_lag**2) + alcohol_lag1 + hr_mean_lag2 + 
                  hr_mean_lag3 + hr_mean_lag1:alcohol_lag1""", data=df_hr_mean).fit()
print(results.summary())
# could plot this

# get corr matrix. what's the multicolinearity? doesn't matter for the 
# covariates? 
def plot_corr_matrix_heatmap(df, var_list, var_list_labels):
    cmap_enter = sns.diverging_palette(15, 125, sep=10, s=70, l=50, as_cmap=True)
    mask = np.zeros_like(df[var_list].corr())
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(df[var_list].corr(), mask=mask, center=0, 
                         square=True, annot=True, fmt='.2f', annot_kws={'size':7}, 
                         cmap=cmap_enter, vmin=-.9, vmax=.9, 
                         xticklabels=var_list_labels, yticklabels=var_list_labels)  #  
 
df_hr_mean.columns
var_list = ['hr_mean', 'alcohol', 'temp', 'month_ts', 'start_sleep_time', 
            'min_asleep', 'subj_sleep', 'energy', 'fun', 'fun_lag1', 
            'subj_sleep_lag1', 'energy_lag1']
plot_corr_matrix_heatmap(df_hr_mean, var_list, var_list)

# plot this relationship between drinks and hr controlling for all vars
# like i did for the industry and wages project. can show ci bars for 
# each alcohol, adjusted for the other variables. it's adjusted means.


# set all drink vars to 0 to get hr when 0 drinks

# for present
# simple model
results = smf.ols(formula = """hr_mean ~ one_drink + two_plus_drinks""", data=df_hr_mean).fit()
print(results.summary())
# get coefs
intercept = results.params[0]
drink_1 = results.params[1]
drink_2 = results.params[2]

y_hr_drinks_0 = intercept + drink_1*0 + drink_2*0 
y_hr_drinks_1 = intercept + drink_1*1 + drink_2*0 
y_hr_drinks_2 = intercept + drink_1*0 + drink_2*1 

se_drinks_0 = results.bse[0]  # this se of the intercept might be se of 0 drinks?
se_drinks_1 = results.bse[1]
se_drinks_2 = results.bse[2]

se_drinks_0 = df_hr_mean[df_hr_mean['alcohol']==0]['hr_mean'].std() / np.sqrt(len(df_hr_mean[df_hr_mean['alcohol']==0]))
se_drinks_1 = df_hr_mean[df_hr_mean['alcohol']==1]['hr_mean'].std() / np.sqrt(len(df_hr_mean[df_hr_mean['alcohol']==1]))
se_drinks_2 = df_hr_mean[df_hr_mean['alcohol']>=2]['hr_mean'].std() / np.sqrt(len(df_hr_mean[df_hr_mean['alcohol']>=2]))

error_bar_list = [1.96*se_drinks_0, 1.96*se_drinks_1, 1.96*se_drinks_2]

#fig = plt.figure(figsize=(12, 5))
plt.bar([0,1,2], [y_hr_drinks_0, y_hr_drinks_1, y_hr_drinks_2], yerr=error_bar_list, color='dodgerblue', 
        alpha=.5, error_kw={'ecolor':'grey', 'linewidth':3, 'ecapsize':30}, align='center')
plt.grid(False)
plt.xticks([0,1,2],['No drinks', '1 drink', '2+ drinks'], fontsize=20)
#plt.yticks([.1,.2,.3,.4],['10%','20%','30%','40%'],fontsize=28)
plt.yticks(fontsize=18)
plt.ylabel('Nightly Heart Rate', fontsize=22)
plt.ylim(45,60)
#plt.text(-.38, .36, 'p < .001', fontsize=30, bbox={'facecolor':'green', 'alpha':0.3, 'pad':5})  
sns.despine()


# look at distribution of hr on nights when drank vs not. 
# wider distrib w alcohol, like the medium post?
df_hr_mean.columns
df_hr_mean[['hr_mean', 'alcohol', 'alcohol_dichot']].corr()

i = 4
df_hr_mean[df_hr_mean['alcohol']==i]['hr_mean'].hist(bins=10, alpha=.6)
plt.grid(False)
plt.xlim(45,70)

df_hr_mean.groupby('alcohol_tertile')['hr_mean'].std() 
df_hr_mean.groupby('alcohol')['hr_mean'].std() 

df_hr_mean.groupby('alcohol_tertile')['hr_mean'].hist(bins=8, 
                  alpha=.5, normed=True)
plt.grid(False)

plt.hist(df_hr_mean.groupby('alcohol_tertile')['hr_mean'].apply(lambda x: x.hist(normed=True, 
         bins=8, alpha=.5)))
plt.grid(False)

sns.violinplot(x='alcohol', y='hr_mean', data=df_hr_mean, alpha=.6)

# all these suggest that after 1 drink the stan dev / variance really does go up
# though haven't done a statistical test. but this might suggest moderators 
# of effects of alcohol on hr. sometimes hr is really high avter 2+ drinks
# and sometimes it's not. the cross-validation heterogeneity test might be
# good to do here.


# for present
# full model
results = smf.ols(formula = """hr_mean ~ one_drink + two_plus_drinks + 
                  tue + wed + thu + fri + sat + sun + temp + month_ts + 
                  start_sleep_time + hr_mean_lag1 + hr_mean_lag2 + 
                  alcohol_lag1 + fun_lag1""", data=df_hr_mean).fit()
print(results.summary())

number_of_ivs_in_model = 2
intercept = results.params[0]
ivs = results.params[1:number_of_ivs_in_model+1].reset_index().rename(columns={0:'coefficient'})
iv_list = list(ivs['index'].values)
iv_coefficient_list = list(ivs['coefficient'].values)
covariates = results.params[number_of_ivs_in_model+1:].reset_index().rename(columns={0:'coefficient'})
covariate_list = list(covariates['index'].values)
covariate_coefficient_list = list(covariates['coefficient'].values)
covariate_mean_list = list(df_hr_mean[covariate_list].mean().values)
covariate_mean_x_coef_list = [coef*mean for coef, mean in zip(covariate_coefficient_list, covariate_mean_list)]

# get mean for each iv
iv_mean_list = []
iv_1_mean = intercept + sum(covariate_mean_x_coef_list)
iv_mean_list.append(iv_1_mean)
for i in range(len(iv_coefficient_list)):
    iv_mean_number = (intercept + sum(covariate_mean_x_coef_list)) + iv_coefficient_list[i]
    iv_mean_list.append(iv_mean_number)

# i think i shold use the std error from the regression
# but what to use for 0 drinks?
se_drinks_0 = df_hr_mean[df_hr_mean['alcohol']==0]['hr_mean'].std() / np.sqrt(len(df_hr_mean[df_hr_mean['alcohol']==0]))
se_drinks_1 = results.bse[1]
se_drinks_2 = results.bse[2]
#se_list = [se_drinks_0, se_drinks_1, se_drinks_2]
error_bar_list = [1.96*se_drinks_0, 1.96*se_drinks_1, 1.96*se_drinks_2]


plt.bar([0,1,2], iv_mean_list, yerr=error_bar_list, color='dodgerblue', 
        alpha=.5, error_kw={'ecolor':'grey', 'linewidth':3, 'ecapsize':30}, align='center')
plt.grid(False)
plt.xticks([0,1,2],['No drinks', '1 drink', '2+ drinks'], fontsize=20)
#plt.yticks([.1,.2,.3,.4],['10%','20%','30%','40%'],fontsize=28)
plt.yticks(fontsize=18)
plt.ylabel('Nightly Heart Rate\n(adjusted for covarirates)', fontsize=22)
plt.ylim(45,60)
#plt.text(-.38, .36, 'p < .001', fontsize=30, bbox={'facecolor':'green', 'alpha':0.3, 'pad':5})  
sns.despine()

results.pvalues.one_drink
results.pvalues.two_plus_drinks

results.tvalues

# effect size 

n_total = len(df_hr_mean[df_hr_mean['one_drink'].notnull()])
n_control = len(df_hr_mean[df_hr_mean['one_drink']==0])
n_treatment = len(df_hr_mean[df_hr_mean['one_drink']==1])
t_value = results.tvalues[1]
cohens_d = t_value * np.sqrt( ((n_treatment + n_control)/(n_treatment * n_control)) * ((n_treatment + n_control) / (n_treatment + n_control - 2) ))

n_total = len(df_hr_mean[df_hr_mean['two_plus_drinks'].notnull()])
n_control = len(df_hr_mean[df_hr_mean['two_plus_drinks']==0])
n_treatment = len(df_hr_mean[df_hr_mean['two_plus_drinks']==1])
t_value = results.tvalues[2]
cohens_d = t_value * np.sqrt( ((n_treatment + n_control)/(n_treatment * n_control)) * ((n_treatment + n_control) / (n_treatment + n_control - 2) ))


len(df_hr_mean['date_sleep'].unique())

#iv_list
#drink_3 = results.params[3]
#drink_4 = results.params[4]
#coef_cov_1 = results.params[5]
# get means

#mean_cov_1 = df_hr_mean['month_ts'].mean()

#y_hr_drinks_0 = intercept + coef_cov_1*mean_cov_1 + drink_1*0 + drink_2*0 + drink_3*0 + drink_4*0 
#y_hr_drinks_1 = intercept + coef_cov_1*mean_cov_1 + drink_1*1 + drink_2*0 + drink_3*0 + drink_4*0 
#y_hr_drinks_2 = intercept + coef_cov_1*mean_cov_1 + drink_1*0 + drink_2*1 + drink_3*0 + drink_4*0 
#y_hr_drinks_3 = intercept + coef_cov_1*mean_cov_1 + drink_1*0 + drink_2*0 + drink_3*1 + drink_4*0 
#y_hr_drinks_4 = intercept + coef_cov_1*mean_cov_1 + drink_1*0 + drink_2*0 + drink_3*0 + drink_4*1 

# get se from the reults. se for coefs
# could get the coefs and compute again with adding the se and substracting the se from coefs to get hr means
se_drinks_0 = results.bse[0]  # this isn't giving the se for 0 drinks, though
se_drinks_1 = results.bse[1]
se_drinks_2 = results.bse[2]
se_drinks_3 = results.bse[3]
se_drinks_4 = results.bse[4]

#se = df_hr_mean[df_hr_mean['alcohol']==0]['hr_mean'].std() / np.sqrt(len(df_hr_mean[df_hr_mean['alcohol']==0]))
#se = df_hr_mean[df_hr_mean['alcohol']==1]['hr_mean'].std() / np.sqrt(len(df_hr_mean[df_hr_mean['alcohol']==1]))
#se = df_hr_mean[df_hr_mean['alcohol']==2]['hr_mean'].std() / np.sqrt(len(df_hr_mean[df_hr_mean['alcohol']==2]))
#se = df_hr_mean[df_hr_mean['alcohol']==3]['hr_mean'].std() / np.sqrt(len(df_hr_mean[df_hr_mean['alcohol']==3]))
#se = df_hr_mean[df_hr_mean['alcohol']==4]['hr_mean'].std() / np.sqrt(len(df_hr_mean[df_hr_mean['alcohol']==4]))


# time series by drinks
df_sleep_8_to_11_resampled['alcohol'] = np.nan
df_sleep_8_to_11_resampled['alcohol'] = df_sleep_8_to_11_resampled['date_sleep'].map(date_to_alcohol_dict)

sns.countplot(x='alcohol', data=df_sleep_8_to_11_resampled, alpha=.6) 

df_sleep_8_to_11_resampled['alcohol'].replace([5,6],[4,4], inplace=True)

fig, ax = plt.subplots(nrows=1, ncols=1)
sns.relplot(x='date_time_bogus', y='hr_clean', 
            data=df_sleep_8_to_11_resampled.sample(n=1000),  # .sample(n=1000), 
            kind='line', ci=95, hue='alcohol', ax=ax)


df_sleep_8_to_11_resampled['alcohol_tertile'] = df_sleep_8_to_11_resampled['alcohol']
df_sleep_8_to_11_resampled['alcohol_tertile'].replace([3,4,5,6],[2,2,2,2], inplace=True)
df_sleep_8_to_11_resampled['alcohol_tertile'].value_counts()
df_sleep_8_to_11_resampled['alcohol_tertile'] = df_sleep_8_to_11_resampled['alcohol_tertile'].astype(str)
df_sleep_8_to_11_resampled['alcohol_tertile'] = df_sleep_8_to_11_resampled['alcohol_tertile'].str.split('.').str[0]
df_sleep_8_to_11_resampled['alcohol_tertile'].replace('nan', np.nan, inplace=True)


# for present
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=80)
#fig, ax = plt.subplots(nrows=1, ncols=1)
sns.relplot(x='date_time_bogus', y='hr_clean', 
            data=df_sleep_8_to_11_resampled,  # .sample(n=1000), 
            kind='line', ci=95, hue='alcohol_tertile', alpha=.5,
            palette=['green', 'orange', 'red'], 
            hue_order=['0','1','2'], ax=ax)
#ax.xaxis_date()
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#axis.xaxis.set_major_locator(HourLocator(byhour))
ax.set_xlabel('Time', fontsize=26)
ax.set_ylabel('Heart Rate', fontsize=26)
#ax.set_yticklabels(list(range(50,70,2)))
yticks = list(range(50,75,5))
yticklabels = [str(tick) for tick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.set_xlim('2018-10-01 22:00:00', '2018-10-02 10:00:00')
ax.set_ylim(48,75)  
ax.legend(['No drinks', '1 drink', '2+ drinks'], fontsize=22)
fig.autofmt_xdate()
#sns.despine()


# =====================================
# save for creating new variables later
df_sleep_8_to_11_resampled.to_pickle('df_sleep_8_to_11_resampled.pkl')
# =====================================
df_sleep_8_to_11_resampled = pd.read_pickle('df_sleep_8_to_11_resampled.pkl')


df_asleep = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['awake']!=1]
df_asleep = df_asleep.sort_values(by='date_time')

# compute minutes from falling alseep and min from waking
# create variable that says how many minutes been aleep for that night
df_asleep['consecutive_numbers'] = np.arange(1.,len(df_asleep)+1)
df_asleep['sum_minutes_asleep'] = df_asleep.groupby(
    'date_sleep')['consecutive_numbers'].transform(lambda x: x.count())
df_asleep['first_consecutive_number_of_group'] = df_asleep.groupby(
    'date_sleep')['consecutive_numbers'].transform(lambda x: x.head(1))
df_asleep['minutes_asleep'] = (df_asleep['consecutive_numbers'] - 
                               df_asleep['first_consecutive_number_of_group'] + 1)
df_asleep['minutes_asleep'] = df_asleep.loc[:, 'minutes_asleep'] - 1
# though tis isn't actually minutes asleep. it's 30-second periods asleep
df_asleep.head()

# count backwards from waking up
df_asleep['minutes_from_waking'] = df_asleep['sum_minutes_asleep'] - df_asleep['minutes_asleep']

# these are in 30-sec intervals. conver to hours
df_asleep['minutes_asleep'] = df_asleep['minutes_asleep'] / 120
df_asleep['minutes_from_waking'] = df_asleep['minutes_from_waking'] / 120

df_asleep['minutes_asleep'].hist()
df_asleep['minutes_from_waking'].hist()

#df_date = df_asleep[df_asleep['date_sleep']=='2017-06-01']


# for present
#fig = plt.figure(figsize=(20, 6), dpi=80)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=80)
sns.relplot(x='minutes_asleep', y='hr_interpolate_clean', 
            data=df_asleep,  # .sample(n=1000), 
            kind='line', ci=95, hue='alcohol_tertile', alpha=.5,
            palette=['green', 'orange', 'red'], 
            hue_order=['0','1','2'], ax=ax)
yticks = list(range(50,75,5))
yticklabels = [str(tick) for tick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.set_xlabel('Number of Hours From Sleep Onset', fontsize=26)
ax.set_ylabel('Heart Rate', fontsize=26)
ax.set_xlim(0,8)
ax.set_ylim(49,71)  
ax.legend(['No drinks', '1 drink', '2+ drinks'], fontsize=22)

#plt.xlim(0,8)
#plt.ylabel('Heart Rate', fontsize=26)
#plt.xlabel('Number of Hours From Sleep Onset', fontsize=26)
#plt.xticks(fontsize=22)
#plt.yticks(fontsize=22)
#plt.legend(['No drinks', '1 drink', '2+ drinks'])
#sns.despine()


# present
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), dpi=80)
sns.relplot(x='minutes_from_waking', y='hr_interpolate_clean', 
            data=df_asleep,  # .sample(n=1000), 
            kind='line', ci=95, hue='alcohol_tertile', alpha=.5,
            palette=['green', 'orange', 'red'], 
            hue_order=['0','1','2'], ax=ax)
yticks = list(range(50,75,5))
yticklabels = [str(tick) for tick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.set_xlabel('Number of Hours From Waking', fontsize=26)
ax.set_ylabel('Heart Rate', fontsize=26)
ax.set_xlim(0,8)
ax.set_ylim(49,71)  
ax.legend(['No drinks', '1 drink', '2+ drinks'], fontsize=22)
ax.invert_xaxis()
#plt.xlim(0,8)
#plt.ylabel('Heart Rate', fontsize=15)
#plt.xlabel('Number of Hours From Waking', fontsize=15)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.gca().invert_xaxis()
#plt.legend(['No drinks', '1 drink', '2+ drinks'])
#sns.despine()


# ==================================
# save for future computing of hr 
# from sleep onset and from waking
df_asleep.to_pickle('df_asleep.pkl')
# ==================================


# explore that lip just before waking. is there a lot of variability in that?
# does that lip appear every time i wake up during the night?
# is it possibe that this is awake time that the fitbit mis-labelled as asleep?
# should look at the raw hr in case the smoothed has a lip because of awake times?
df_asleep.shape
df_asleep.head()
df_asleep.tail()
df_asleep['minutes_from_waking'].hist(alpha=.75)
plt.grid(False)

df_asleep_final_hour = df_asleep[df_asleep['minutes_from_waking']<.5]
#df_asleep_final_hour = df_asleep[df_asleep['minutes_from_waking']<60]
df_asleep_final_hour.columns

# overlay many nights
alpha_level = .3
width_for_line = 3
yticks = list(range(40,80,5))
yticklabels = [str(tick) for tick in yticks]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), dpi=80)
for date in df_asleep_final_hour['date_sleep'].unique()[30:35]:
    df_date = df_asleep_final_hour[df_asleep_final_hour['date_sleep']==date]
    plt.plot(df_date['minutes_from_waking'], df_date['hr_interpolate_clean'],  # hr_interpolate_clean is appropriate here. it's basically raw data. hr_rolling_30_min
             alpha=alpha_level, linewidth=width_for_line)  # color='green', 
ax.set_xlabel('Number of Hours From Waking', fontsize=26)
ax.set_ylabel('Heart Rate', fontsize=26)
ax.set_xlim(0,.5)
ax.set_ylim(43,72)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.invert_xaxis()
sns.despine()


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), dpi=80)
sns.relplot(x='minutes_from_waking', y='hr_interpolate_clean', 
            data=df_asleep_final_hour,  # .sample(n=1000), 
            kind='line', ci=95, alpha=.5, ax=ax)
yticks = list(range(50,65,5))
yticklabels = [str(tick) for tick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
ax.set_xlabel('Number of Hours From Waking', fontsize=26)
ax.set_ylabel('Heart Rate', fontsize=26)
ax.set_xlim(0,.1)
ax.set_ylim(49,61)  
ax.invert_xaxis()

# i guess this plot above suggests that there really is a lip starting 
# at about .015. that's a little under 1 minute. So could take the final
# minute as "the lip" and could cut it off and/or use it as a variable.

# but that's where the avg lip starts. there's variation.
# .1 is 6 min before waking. and sometimes the lip starts then -
# i.e., the quick increase in hr starts then. seems that if fitbit
# is mislabeling awake vs. asleep here, that would only see the spike
# in hr for the final 1-2 mins. this suggests to me that the increase
# in hr that is this "lip" is happening when I'm asleep. but actually, 
# still can't be sure. i still could be just lying there awake for 5-10 
# min but so still that the fitbit things I'm asleep. either way, 
# I like the idea of usig the size of this lip as an IV to see if it
# predict things.

# when i look at just a handful of timeseries in isolation, the lips
# look less like an artifcact and more like something real.

# unless i do a little experiment, and note the exact time i wake
# for a couple weeks. and then compare to what fitbit says, i can't
# tell right now if these final moments in the sleep data in which 
# i get this lip -- this fast incrase in hr -- is happening when i'm 
# alseep or because i'm waking and fitbit is labelling it as me being
# asleep still. it's just a tiny bit of data, so it doesn't matter
# at the moment for anys. But wouldn't hurt to include the size of the
# lip as an IV to see if this means anything for health the next day
# or sleep quality. 

# like the idea of doing this experiment. if the lip is just an artifact
# of the fitbit asleep-awake labelling algo, then the lip isn't really
# interesting. but if it's not and the lip is happening when i'm asleep,
# this is a phenoenon that's fascinating to me. it means that in my sleep,
# while i'm unconscsious, i'm quickly getting ready to wake.


# does type of alcohol matter?
#df_asleep['alcohol_type'] = df_asleep['date_sleep'].map(date_to_alcohol_type_dict)
#df_asleep_alcohol = df_asleep[df_asleep['alcohol']==1]
#df_asleep_alcohol['alcohol_type'].value_counts()
#df_asleep_alcohol = df_asleep_alcohol[df_asleep_alcohol['alcohol_type'].isin(['Beer', 'Red wine', 'Hard alcohol'])]
#
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), dpi=80)
#sns.relplot(x='minutes_asleep', y='hr_interpolate_clean', 
#            data=df_asleep_alcohol,  # .sample(n=1000), 
#            kind='line', ci=90, hue='alcohol_type', alpha=.5,
#            #palette=['orange', 'red'], 
#            #hue_order=['1','2'], 
#            ax=ax)
#yticks = list(range(50,75,5))
#yticklabels = [str(tick) for tick in yticks]
#ax.set_yticks(yticks)
#ax.set_yticklabels(yticklabels)
#ax.xaxis.set_tick_params(labelsize=22)
#ax.yaxis.set_tick_params(labelsize=22)
#ax.set_xlabel('Number of Hours From Sleep Onset', fontsize=26)
#ax.set_ylabel('Heart Rate', fontsize=26)
#ax.set_xlim(0,8)
#ax.set_ylim(49,71) 
#ax.legend(fontsize=22) 
##ax.legend(['1 drink', '2+ drinks'], fontsize=22)
#
## do in regression
#df_asleep['alcohol_type'].unique()
#
#results = smf.ols(formula = """hr_mean ~ alcohol +
#                  tue + wed + thu + fri + sat + sun + temp + month_ts + 
#                  start_sleep_time + hr_mean_lag1 + hr_mean_lag2 + 
#                  alcohol_lag1 + fun_lag1""", data=df_hr_mean).fit()
#print(results.summary())
#
#df_asleep['alcohol_type'].replace(np.nan, 'none', inplace=True)
#df_asleep['alcohol_type'].unique()
#df_asleep['beer'] = 0
#df_asleep.loc[df_asleep['alcohol_type'].str.contains('Beer'), 'beer'] = 1
#df_asleep['red_wine'] = 0
#df_asleep.loc[df_asleep['alcohol_type'].str.contains('Red wine'), 'red_wine'] = 1
#df_asleep['white_wine'] = 0
#df_asleep.loc[df_asleep['alcohol_type'].str.contains('White wine'), 'white_wine'] = 1
#df_asleep['mixed_drink'] = 0
#df_asleep.loc[df_asleep['alcohol_type'].str.contains('Mixed drink'), 'mixed_drink'] = 1
#df_asleep['hard_alcohol'] = 0
#df_asleep.loc[df_asleep['alcohol_type'].str.contains('Hard alcohol'), 'hard_alcohol'] = 1
#df_asleep['champaigne'] = 0
#df_asleep.loc[df_asleep['alcohol_type'].str.contains('Champaigne'), 'champaigne'] = 1
#
#for alcohol in ['beer', 'red_wine', 'white_wine', 'mixed_drink', 'hard_alcohol', 'champaigne']:
#    print(df_asleep[alcohol].value_counts(normalize=True))
#    print()
#
#df_hr_mean['alcohol_type'] = df_hr_mean['date_sleep'].map(date_to_alcohol_type_dict)
#df_hr_mean['alcohol_type'].replace(np.nan, 'none', inplace=True)
#df_hr_mean['beer'] = 0
#df_hr_mean.loc[df_hr_mean['alcohol_type'].str.contains('Beer'), 'beer'] = 1
#df_hr_mean['red_wine'] = 0
#df_hr_mean.loc[df_hr_mean['alcohol_type'].str.contains('Red wine'), 'red_wine'] = 1
#df_hr_mean['white_wine'] = 0
#df_hr_mean.loc[df_hr_mean['alcohol_type'].str.contains('White wine'), 'white_wine'] = 1
#df_hr_mean['mixed_drink'] = 0
#df_hr_mean.loc[df_hr_mean['alcohol_type'].str.contains('Mixed drink'), 'mixed_drink'] = 1
#df_hr_mean['hard_alcohol'] = 0
#df_hr_mean.loc[df_hr_mean['alcohol_type'].str.contains('Hard alcohol'), 'hard_alcohol'] = 1
#df_hr_mean['champaigne'] = 0
#df_hr_mean.loc[df_hr_mean['alcohol_type'].str.contains('Champaigne'), 'champaigne'] = 1
#
#for alcohol in ['beer', 'red_wine', 'white_wine', 'mixed_drink', 'hard_alcohol', 'champaigne']:
#    print(df_hr_mean[alcohol].value_counts(normalize=True))
#    print()
#
#results = smf.ols(formula = """hr_mean ~ alcohol +
#                  tue + wed + thu + fri + sat + sun + temp + month_ts + 
#                  start_sleep_time + hr_mean_lag1 + hr_mean_lag2 + 
#                  alcohol_lag1 + fun_lag1 + beer + red_wine + 
#                  white_wine + mixed_drink + hard_alcohol""", data=df_hr_mean[df_hr_mean['alcohol']>0]).fit()
#print(results.summary())
#
#df_hr_mean_alcohol_1 = df_hr_mean[df_hr_mean['alcohol']==1]
#
#results = smf.ols(formula = """hr_mean ~
#                  tue + wed + thu + fri + sat + sun + temp + month_ts + 
#                  start_sleep_time + hr_mean_lag1 + hr_mean_lag2 + 
#                  alcohol_lag1 + fun_lag1 + beer + red_wine + 
#                  hard_alcohol""", data=df_hr_mean_alcohol_1).fit()
#print(results.summary())
#
#df_hr_mean_alcohol_1['alcohol_type_1_drink'] = 'other'
#df_hr_mean_alcohol_1.loc[df_hr_mean_alcohol_1['red_wine']==1, 'alcohol_type_1_drink'] = 'red_wine'
#df_hr_mean_alcohol_1.loc[df_hr_mean_alcohol_1['hard_alcohol']==1, 'alcohol_type_1_drink'] = 'hard_alcohol'
#df_hr_mean_alcohol_1.loc[df_hr_mean_alcohol_1['beer']==1, 'alcohol_type_1_drink'] = 'beer'
#
#sns.barplot(x='alcohol_type_1_drink', y='hr_mean', data=df_hr_mean_alcohol_1, 
#            color='dodgerblue', alpha=.7)
#plt.ylim(40,57)
# raw means show that hard alcohol may be worst for hr
# but when control for all other vars above, no difference statistically


# -------------------------------------------------------------------
# incorp resting hr measures to help validate sleep quality metric(s)

## save df with sedentary hr mean ea day as measure of resting hr
## computed by getting the mean sedentary hr each hour and then taking
## the average of these hourly measurements
df_sedentary_hr_mean_day = pd.read_pickle('df_sedentary_hr_mean_day.pkl')
## save df with minimum sedentary hr mean ea day as measure of resting hr
## computed by getting the minium sedentary hr each hour and then taking
## the average of these hourly measurements
df_minimum_hr_mean_day = pd.read_pickle('df_minimum_hr_mean_day.pkl')

date_to_hr_resting_dict = dict(df_sedentary_hr_mean_day)
date_to_hr_resting_minimum_dict = dict(df_minimum_hr_mean_day)

df_hr_mean.head()
df_hr_mean.columns

df_hr_mean['hr_rest'] = df_hr_mean['date_sleep'].map(date_to_hr_resting_dict)
df_hr_mean['hr_rest_min'] = df_hr_mean['date_sleep'].map(date_to_hr_resting_minimum_dict)

df_hr_mean[df_hr_mean['hr_rest'].isnull()]
df_hr_mean[df_hr_mean['hr_rest_min'].isnull()]
# just 5 days missing. seem like random days. ok.

# create future lagged resting hr -- i.e., resting hr next day
def create_lagged_future_date_variables(number_of_lags, date_variable, df):
    for lag in list(range(1,number_of_lags+1)):
        df['date_forward_'+str(lag)] = df[date_variable] + timedelta(days=lag)
    return df

df_hr_mean = create_lagged_future_date_variables(1, 'date_sleep', df_hr_mean)
df_hr_mean.head()

df_hr_mean['hr_rest_next_day'] = df_hr_mean['date_forward_1'].map(date_to_hr_resting_dict)
df_hr_mean['hr_rest_min_next_day'] = df_hr_mean['date_forward_1'].map(date_to_hr_resting_minimum_dict)

df_hr_mean[['date_sleep', 'hr_rest', 'hr_rest_next_day', 'hr_mean']].corr()
# raw corr suggesting that sleeping hr might be more affected by prior 
# day hr than it affects next day hr?! though both high corrs.

plt.scatter(x='hr_mean', y='hr_rest_next_day', data=df_hr_mean, alpha=.25)

sns.lmplot(x='hr_mean', y='hr_rest_next_day', data=df_hr_mean, 
           scatter_kws={'alpha':.25}, x_partial='hr_rest', y_partial='hr_rest')

df_hr_mean.columns

# =========
# LEFT OFF
#df_hr_mean.to_pickle('df_hr_mean_w_resting_hr.pkl')
# =========
df_hr_mean = pd.read_pickle('df_hr_mean_w_resting_hr.pkl')
df_hr_mean.shape
df_hr_mean.columns
df_hr_mean.head()

len(df_hr_mean[(df_hr_mean['hr_mean_lag1'].notnull()) & 
               (df_hr_mean['hr_mean_lag2'].notnull()) & 
               (df_hr_mean['hr_mean_lag3'].notnull()) & 
               (df_hr_mean['hr_rest_next_day'].notnull())])
# all the lags drastically cut down on sample size. impute?
# should the appraoch to imputing depend on the particular anys?
# for now, just anys when there's data an don't impute
df_hr_mean[['date_sleep', 'hr_mean', 'hr_mean_lag1', 'hr_rest_next_day']]

sns.relplot(x='date_sleep', y='hr_mean_lag1', data=df_hr_mean[df_hr_mean.index<50], kind='line')
dftest = df_hr_mean[df_hr_mean.index<50]
plt.plot(dftest['date_sleep'], dftest['hr_mean_lag1'])
plt.xticks(rotation=30)


# ----------    
# examine relationship between resting hr and resting hr the next day
# cubic? or something like that? because if high resting hr, it's easier
# to reduce it the next day, vs. if resting hr is already low.
plt.scatter(df_hr_mean['hr_rest'], 
            df_hr_mean['hr_rest_next_day'], alpha=.25, color='dodgerblue')

sns.lmplot(x='hr_rest', y='hr_rest_next_day', 
           data=df_hr_mean, scatter_kws={'alpha':.25, 'color':'dodgerblue'},
           order=3)
plt.xlim(50,80)
plt.ylim(50,80)

sns.lmplot(x='hr_rest', y='hr_rest_next_day', 
           data=df_hr_mean, scatter_kws={'alpha':.25, 'color':'dodgerblue'},
           x_bins=20)


plt.plot(df_hr_mean['hr_rest'].rolling(30).mean(), alpha=.6)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# kind of saying a couple things. 1 - looks like when hr today is low,
# that the body is pretty stable because don't tend to get huge changes
# in hr the next day, even in the upward direction. can see that the
# variance around the regression line is smaller on the far left side.
# that might be pretty cool -- that at some point when healthy, it's not
# too easily undone. on right side, with high hr today, see lot of variability
# for hr the next day. come back to this. ideally, shouldn't i get the 
# distribution of hr over the regession line (like something raaz was
# working on) and score hr the next day based on the pctile of where in 
# that distirbution the next day's hr falls? because with the way the
# data is now, i think i'll be examining largely whether good sleep 
# reduces hr when resting hr was mid to high the previous day. but
# what about when hr was low the prior day? what's the effect of a 
# good night sleep? not sure i'll be able to tell the way the data is now.

# think i should use cubit to model the lag, or at least quadratic
# because can see it fits the data much better
results = smf.ols(formula = """hr_rest_next_day ~ hr_rest""",
                  data=df_hr_mean).fit()  
print(results.summary())  # adj r sq = 0.082

results = smf.ols(formula = """hr_rest_next_day ~ hr_rest + I(hr_rest**2) + I(hr_rest**3)""",
                  data=df_hr_mean).fit()  
print(results.summary())  # adj r sq = 0.101

results = smf.ols(formula = """hr_rest_next_day ~ hr_rest + I(hr_rest**2)""",
                  data=df_hr_mean).fit()  
print(results.summary())  # adj r sq = 0.103

results = smf.ols(formula = """hr_rest_next_day ~ hr_rest + I(hr_rest**2) + 
                  I(hr_rest**3) + I(hr_rest**4)""",
                  data=df_hr_mean).fit()  
print(results.summary())  # adj r sq = 0.103

# quadratic definitelyl fits better than just the linear

#df_hr_mean = df_hr_mean[(df_hr_mean['hr_rest'].notnull()) & (df_hr_mean['hr_rest_next_day'].notnull())]
#df_hr_mean['hr_rest_round'] = df_hr_mean['hr_rest'].round(0)
#df_hr_mean['hr_rest_round'] = df_hr_mean['hr_rest_round'].astype(int)

# bin 'hr_rest'
df_hr_mean['hr_rest'].min()
df_hr_mean['hr_rest'].max()

#custom_bucket_array = custom_bucket_array = np.linspace(53, 81, 2)
def plot_relationship_with_bins(df_hr_mean, bin_size):
    custom_bucket_array = np.arange(52, 82, bin_size)
    df_hr_mean['hr_rest_bins'] = pd.cut(df_hr_mean['hr_rest'], custom_bucket_array)
    sns.barplot(x='hr_rest_bins', y='hr_rest_next_day', data=df_hr_mean, 
                color='dodgerblue', alpha=.6, errcolor='grey')
    plt.xticks(rotation=60)
    plt.ylim(40,70)
    
plot_relationship_with_bins(df_hr_mean, 4)
plot_relationship_with_bins(df_hr_mean, 3)
# not the s-shape I was expecting. Yes, the higher the prior day's
# resting hr, the more likely the next day's is going to be lower.
# but lower resting hr doesn't predict an increasing likelihood of
# next day's hr being higher.
plot_relationship_with_bins(df_hr_mean, 2)
plot_relationship_with_bins(df_hr_mean, 1)
plot_relationship_with_bins(df_hr_mean, .5)
plt.ylim(50,70)

# i should try logging the diffs between each day and the next
# to "stationarize" the data and would that take into account
# large differences emerging from certain ranges of day 1 hr?
#results = smf.ols(formula = 'hr_rest_next_day ~ hr_rest', data=df_hr_mean).fit()
#print(results.summary())
#plt.hist(results.resid)

# EXAMINE SLEEP QUALITY METIRICS (HR-WHILE-SLEEPING AND OTHERS) ON RESTING HR
# CONTROLLING FOR LAGGED VARS (ACCOUNTING FOR THAT CURVILINEAR RLEATION BETWEEN
# PRIOR RESITNG HR AND FUTURE RESTING HR) AND CONFOUNDS.

df_hr_mean.columns

results = smf.ols(formula = 'hr_rest_next_day ~ hr_rest', data=df_hr_mean).fit()
print(results.summary())

results = smf.ols(formula = """hr_rest_next_day ~ hr_rest + I(hr_rest**2) + 
                  hr_mean + hr_mean_lag1 + temp + month_ts + start_sleep_time +  
                  tue + wed + thu + fri + sat + sun""",  #  + alcohol + alcohol_lag1 
                  data=df_hr_mean).fit()
print(results.summary())
# hr_mean, which is sleeping hr is the biggest driver here
# + I(hr_rest**2) + I(hr_rest**3)
# hr_rest + 
# hr_rest_min
# hr_rest_next_day
# hr_rest_min_next_day
#results = smf.ols(formula = """hr_rest_next_day ~ hr_mean""",  #  + alcohol + alcohol_lag1 
#                  data=df_hr_mean).fit()
#print(results.summary())

# look at other sleep metrics - 
# general approach -- get date to metric dictionary
# then can map onto lagged version of the date to get 
# lagged variables even if there are gaps in the dates/timeseries

# min_asleep = number of minutes asleep that night
date_to_minutes_alseep_dict = dict(zip(df_hr_mean['date_sleep'], df_hr_mean['min_asleep']))

# get times awake during sleep period (sort of an efficicncy of sleep)
# this dict has the minutes between start and end points of sleep 
# and so includes times in between when i woke up
with open('date_to_min_within_sleep_dict.pkl', 'rb') as picklefile:
    date_to_min_within_sleep_dict = pickle.load(picklefile)

df_hr_mean['minutes_within_sleep'] = df_hr_mean['date_sleep'].map(date_to_min_within_sleep_dict)

# both may be important
df_hr_mean[['minutes_within_sleep', 'min_asleep']].corr()


# create new sleep metrics from min-by-min data
df_sleep_8_to_11_resampled = pd.read_pickle('df_sleep_8_to_11_resampled.pkl')
df_sleep_8_to_11_resampled.shape
df_sleep_8_to_11_resampled.columns

df_sleep_8_to_11_resampled[['date_time', 'awake', 'restless', 'hr_clean']]
# look at restless over time
df_sleep_8_to_11_resampled[['start_of_date_sleep', 'start_sleep_occurs_before', 
                            'sleep_session', 'start_sleep', 'end_sleep']].head()
df_sleep_8_to_11_resampled[['start_of_date_sleep', 'start_sleep_occurs_before', 
                            'sleep_session', 'start_sleep', 'end_sleep']].tail()

# get amount of rem and deep and light sleep each night
# then get hr for each. and save dicts.

df_sleep_8_to_11_resampled['deep']
df_sleep_8_to_11_resampled.groupby('date_sleep')['deep'].apply(lambda x: sum(x))
# and divide these by 2 to get minutes

# if nans, it sums to np.nan. so fill nans with 0s
df_sleep_8_to_11_resampled['deep'].replace(np.nan, 0, inplace=True)

df_deep = df_sleep_8_to_11_resampled.groupby('date_sleep')['deep'].apply(lambda x: sum(x)).reset_index()
df_deep = df_deep[df_deep['deep']>0]
df_deep['deep'] = df_deep['deep'] / 2
df_deep.hist(alpha=.75)
plt.grid(False)

date_sleep_to_deep_minutes_dict = dict(zip(df_deep['date_sleep'], df_deep['deep']))
# save dict
with open('date_sleep_to_deep_minutes_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_deep_minutes_dict, picklefile)


# if nans, it sums to np.nan. so fill nans with 0s
df_sleep_8_to_11_resampled['rem'].replace(np.nan, 0, inplace=True)
df_rem = df_sleep_8_to_11_resampled.groupby('date_sleep')['rem'].apply(lambda x: sum(x)).reset_index()

df_rem = df_rem[df_rem['rem']>0]
df_rem['rem'] = df_rem['rem'] / 2
df_rem.hist(alpha=.75)
plt.grid(False)

date_sleep_to_rem_minutes_dict = dict(zip(df_rem['date_sleep'], df_rem['rem']))
# save dict
with open('date_sleep_to_rem_minutes_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_rem_minutes_dict, picklefile)

plt.hist(df_rem['rem'], alpha=.35)
plt.hist(df_deep['deep'], alpha=.35)
plt.grid(False)


# if nans, it sums to np.nan. so fill nans with 0s
df_sleep_8_to_11_resampled['light'].replace(np.nan, 0, inplace=True)
df_light = df_sleep_8_to_11_resampled.groupby('date_sleep')['light'].apply(lambda x: sum(x)).reset_index()

df_light = df_light[df_light['light']>0]
df_light['light'] = df_light['light'] / 2
df_light.hist(alpha=.75)
plt.grid(False)

date_sleep_to_light_minutes_dict = dict(zip(df_light['date_sleep'], df_light['light']))
# save dict
with open('date_sleep_to_light_minutes_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_light_minutes_dict, picklefile)


# if nans, it sums to np.nan. so fill nans with 0s
df_sleep_8_to_11_resampled['restless'].replace(np.nan, 0, inplace=True)
df_restless = df_sleep_8_to_11_resampled.groupby('date_sleep')['restless'].apply(lambda x: sum(x)).reset_index()

df_restless = df_restless[df_restless['restless']>0]
df_restless['restless'] = df_restless['restless'] / 2
df_restless.hist(alpha=.75)
plt.grid(False)

date_sleep_to_restless_minutes_dict = dict(zip(df_restless['date_sleep'], df_restless['restless']))
# save dict
with open('date_sleep_to_restless_minutes_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_restless_minutes_dict, picklefile)

# look at over time
plt.plot(df_restless['date_sleep'], df_restless['restless'].rolling(window=30).mean())
# has it changed much over time? maybe ok to use?


# ------
# create smoothness/variability of hr for each night metrics
df_hr_notnull = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['hr_clean'].notnull()]
df_std_daily = df_hr_notnull.groupby('date_sleep')['hr_clean'].apply(lambda x: x.std()).reset_index()
plt.plot(df_std_daily['date_sleep'], df_std_daily['hr_clean'].rolling(window=30).mean())
date_sleep_to_hr_std_dict = dict(zip(df_std_daily['date_sleep'], df_std_daily['hr_clean']))
# save dict
with open('date_sleep_to_hr_std_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_hr_std_dict, picklefile)
# pretty sure std and mean are correlated. adj for mean?
# other ways to get smoothness? within certain frequencies?


# ------
# hr in sleep stages
df_deep_hr = df_hr_notnull.groupby(['date_sleep', 'deep'])['hr_clean'].apply(lambda x: np.mean(x)).reset_index()
df_deep_hr = df_deep_hr[df_deep_hr['deep']==1]
df_deep_hr = df_deep_hr[df_deep_hr['hr_clean']>0]
df_deep_hr['hr_clean'].hist(alpha=.75)
plt.grid(False)
plt.axvline(df_deep_hr['hr_clean'].mean(), linestyle='--', alpha=.5, color='black')

date_sleep_to_deep_hr_dict = dict(zip(df_deep_hr['date_sleep'], df_deep_hr['hr_clean']))
# save dict
with open('date_sleep_to_deep_hr_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_deep_hr_dict, picklefile)

df_deep_hr_std = df_hr_notnull.groupby(['date_sleep', 'deep'])['hr_clean'].apply(lambda x: np.std(x)).reset_index()
df_deep_hr_std = df_deep_hr_std[df_deep_hr_std['deep']==1]
df_deep_hr_std = df_deep_hr_std[df_deep_hr_std['hr_clean']>0]
df_deep_hr_std['hr_clean'].hist(alpha=.75)
plt.grid(False)
plt.axvline(df_deep_hr_std['hr_clean'].mean(), linestyle='--', alpha=.5, color='black')

date_sleep_to_deep_hr_std_dict = dict(zip(df_deep_hr_std['date_sleep'], df_deep_hr_std['hr_clean']))
# save dict
with open('date_sleep_to_deep_hr_std_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_deep_hr_std_dict, picklefile)

df_deep = df_hr_notnull[df_hr_notnull['deep']==1]
df_deep = df_deep.sort_values(by='date_time')
df_deep['hr_deep_ewm_from_wake'] = df_deep.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(span=10, min_periods=10).mean())
df_deep_hr_ewm_from_wake = df_deep.groupby('date_sleep')['hr_deep_ewm_from_wake'].apply(lambda x: x.tail(1)).reset_index()
df_deep_hr_ewm_from_wake['hr_deep_ewm_from_wake'].hist(alpha=.75)
plt.grid(False)
date_sleep_to_deep_hr_ewm_from_wake_dict = dict(zip(df_deep_hr_ewm_from_wake['date_sleep'], df_deep_hr_ewm_from_wake['hr_deep_ewm_from_wake']))
# save dict
with open('date_sleep_to_deep_hr_ewm_from_wake_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_deep_hr_ewm_from_wake_dict, picklefile)

df_deep = df_deep.sort_values(by='date_time', ascending=False)
df_deep['hr_deep_ewm_from_onset'] = df_deep.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(span=10, min_periods=10).mean())
df_deep_hr_ewm_from_onset = df_deep.groupby('date_sleep')['hr_deep_ewm_from_onset'].apply(lambda x: x.tail(1)).reset_index()
df_deep_hr_ewm_from_onset['hr_deep_ewm_from_onset'].hist(alpha=.75)
plt.grid(False)
date_sleep_to_deep_hr_ewm_from_onset_dict = dict(zip(df_deep_hr_ewm_from_onset['date_sleep'], df_deep_hr_ewm_from_onset['hr_deep_ewm_from_onset']))
# save dict
with open('date_sleep_to_deep_hr_ewm_from_onset_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_deep_hr_ewm_from_onset_dict, picklefile)


df_rem_hr = df_hr_notnull.groupby(['date_sleep', 'rem'])['hr_clean'].apply(lambda x: np.mean(x)).reset_index()
df_rem_hr = df_rem_hr[df_rem_hr['rem']==1]
df_rem_hr = df_rem_hr[df_rem_hr['hr_clean']>0]
df_rem_hr['hr_clean'].hist(alpha=.75)
plt.grid(False)
plt.axvline(df_rem_hr['hr_clean'].mean(), linestyle='--', alpha=.5, color='black')

date_sleep_to_rem_hr_dict = dict(zip(df_rem_hr['date_sleep'], df_rem_hr['hr_clean']))
# save dict
with open('date_sleep_to_rem_hr_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_rem_hr_dict, picklefile)

df_rem_hr_std = df_hr_notnull.groupby(['date_sleep', 'rem'])['hr_clean'].apply(lambda x: np.std(x)).reset_index()
df_rem_hr_std = df_rem_hr_std[df_rem_hr_std['rem']==1]
df_rem_hr_std = df_rem_hr_std[df_rem_hr_std['hr_clean']>0]
df_rem_hr_std['hr_clean'].hist(alpha=.75)
plt.grid(False)
plt.axvline(df_rem_hr_std['hr_clean'].mean(), linestyle='--', alpha=.5, color='black')

date_sleep_to_rem_hr_std_dict = dict(zip(df_rem_hr_std['date_sleep'], df_rem_hr_std['hr_clean']))
# save dict
with open('date_sleep_to_rem_hr_std_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_rem_hr_std_dict, picklefile)

df_rem = df_hr_notnull[df_hr_notnull['rem']==1]
df_rem = df_rem.sort_values(by='date_time')
df_rem['hr_rem_ewm_from_wake'] = df_rem.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(span=10, min_periods=10).mean())
df_rem_hr_ewm_from_wake = df_rem.groupby('date_sleep')['hr_rem_ewm_from_wake'].apply(lambda x: x.tail(1)).reset_index()
df_rem_hr_ewm_from_wake['hr_rem_ewm_from_wake'].hist(alpha=.75)
plt.grid(False)
date_sleep_to_rem_hr_ewm_from_wake_dict = dict(zip(df_rem_hr_ewm_from_wake['date_sleep'], df_rem_hr_ewm_from_wake['hr_rem_ewm_from_wake']))
# save dict
with open('date_sleep_to_rem_hr_ewm_from_wake_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_rem_hr_ewm_from_wake_dict, picklefile)

df_rem = df_rem.sort_values(by='date_time', ascending=False)
df_rem['hr_rem_ewm_from_onset'] = df_rem.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(span=10, min_periods=10).mean())
df_rem_hr_ewm_from_onset = df_rem.groupby('date_sleep')['hr_rem_ewm_from_onset'].apply(lambda x: x.tail(1)).reset_index()
df_rem_hr_ewm_from_onset['hr_rem_ewm_from_onset'].hist(alpha=.75)
plt.grid(False)
date_sleep_to_rem_hr_ewm_from_onset_dict = dict(zip(df_rem_hr_ewm_from_onset['date_sleep'], df_rem_hr_ewm_from_onset['hr_rem_ewm_from_onset']))
# save dict
with open('date_sleep_to_rem_hr_ewm_from_onset_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_rem_hr_ewm_from_onset_dict, picklefile)


df_light_hr = df_hr_notnull.groupby(['date_sleep', 'light'])['hr_clean'].apply(lambda x: np.mean(x)).reset_index()
df_light_hr = df_light_hr[df_light_hr['light']==1]
df_light_hr = df_light_hr[df_light_hr['hr_clean']>0]
df_light_hr['hr_clean'].hist(alpha=.75)
plt.grid(False)
plt.axvline(df_light_hr['hr_clean'].mean(), linestyle='--', alpha=.5, color='black')

date_sleep_to_light_hr_dict = dict(zip(df_light_hr['date_sleep'], df_light_hr['hr_clean']))
# save dict
with open('date_sleep_to_light_hr_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_light_hr_dict, picklefile)

df_light_hr_std = df_hr_notnull.groupby(['date_sleep', 'light'])['hr_clean'].apply(lambda x: np.std(x)).reset_index()
df_light_hr_std = df_light_hr_std[df_light_hr_std['light']==1]
df_light_hr_std = df_light_hr_std[df_light_hr_std['hr_clean']>0]
df_light_hr_std['hr_clean'].hist(alpha=.75)
plt.grid(False)
plt.axvline(df_light_hr_std['hr_clean'].mean(), linestyle='--', alpha=.5, color='black')

date_sleep_to_light_hr_std_dict = dict(zip(df_light_hr_std['date_sleep'], df_light_hr_std['hr_clean']))
# save dict
with open('date_sleep_to_light_hr_std_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_light_hr_std_dict, picklefile)

df_light = df_hr_notnull[df_hr_notnull['light']==1]
df_light = df_light.sort_values(by='date_time')
df_light['hr_light_ewm_from_wake'] = df_light.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(span=10, min_periods=10).mean())
df_light_hr_ewm_from_wake = df_light.groupby('date_sleep')['hr_light_ewm_from_wake'].apply(lambda x: x.tail(1)).reset_index()
df_light_hr_ewm_from_wake['hr_light_ewm_from_wake'].hist(alpha=.75)
plt.grid(False)
date_sleep_to_light_hr_ewm_from_wake_dict = dict(zip(df_light_hr_ewm_from_wake['date_sleep'], df_light_hr_ewm_from_wake['hr_light_ewm_from_wake']))
# save dict
with open('date_sleep_to_light_hr_ewm_from_wake_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_light_hr_ewm_from_wake_dict, picklefile)

df_light = df_light.sort_values(by='date_time', ascending=False)
df_light['hr_light_ewm_from_onset'] = df_light.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(span=10, min_periods=10).mean())
df_light_hr_ewm_from_onset = df_light.groupby('date_sleep')['hr_light_ewm_from_onset'].apply(lambda x: x.tail(1)).reset_index()
df_light_hr_ewm_from_onset['hr_light_ewm_from_onset'].hist(alpha=.75)
plt.grid(False)
date_sleep_to_light_hr_ewm_from_onset_dict = dict(zip(df_light_hr_ewm_from_onset['date_sleep'], df_light_hr_ewm_from_onset['hr_light_ewm_from_onset']))
# save dict
with open('date_sleep_to_light_hr_ewm_from_onset_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_sleep_to_light_hr_ewm_from_onset_dict, picklefile)


plt.hist(df_rem_hr['hr_clean'], alpha=.35, color='orange', label='rem hr')
plt.hist(df_deep_hr['hr_clean'], alpha=.35, color='dodgerblue', label='deep hr')
plt.grid(False)
plt.legend()

plt.hist(df_rem_hr['hr_clean'], alpha=.35, color='orange', label='rem hr')
plt.hist(df_light_hr['hr_clean'], alpha=.35, color='dodgerblue', label='light hr')
plt.grid(False)
plt.legend()

plt.hist(df_deep_hr['hr_clean'], alpha=.35, color='dodgerblue', label='deep hr')
plt.hist(df_light_hr['hr_clean'], alpha=.35, color='orange', label='light hr')
plt.grid(False)
plt.legend()

df_deep_hr['hr_clean'].mean()
df_light_hr['hr_clean'].mean()


# ------
# hr at diff points in the night - early, mid, late, etc.
# how to measure? 
# hr first hour, hr last hour
# better -- weighted ewm from start and weighted ewm from end
# though might worry about that hr lip from waking? how to deal with that?
# cut off the final 15 min before waking? 
# but also compute hr in that final 15 min as separate metric.
# can get the data from those plots from sleep onset and waking graphs
# hr first hour, hr second hour, hr third hour, etc. but this could
# be weird and could be weird in conjuction with sleep stages.
# could map out my own sleep stages and get at rem/deep occurances
# e.g., deep/rem hr first occurance, deep/rem hr second occurrence, etc.
# that would get at hr at diff points in the night while controlling
# for sleep stage.  
df_hr_notnull.head()
 
df_asleep = pd.read_pickle('df_asleep.pkl')
df_asleep.head()
df_asleep[['date_time', 'hr_clean']]
df_asleep = df_asleep[df_asleep['hr_clean'].notnull()]
df_asleep.columns
df_asleep[['date_time', 'hr_clean', 'minutes_from_waking', 'minutes_asleep']]
# guess the min from sleep onset and from waking dont matter?
# for computing ewm from sleep onset and from waking? figout out 
# how and look up ewm.

df_asleep = df_asleep.sort_values(by='date_time')
df_asleep['hr_ewm_span_60_from_wake'] = df_asleep.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(span=60, min_periods=180).mean())
df_asleep['hr_ewm_halflife_120_from_wake'] = df_asleep.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(halflife=120, min_periods=180).mean())
df_asleep[['hr_clean', 'hr_ewm_span_60_from_wake', 'hr_ewm_halflife_120_from_wake']].tail()

df_asleep_reversed = df_asleep.sort_values(by='date_time', ascending=False)
df_asleep['date_time']
df_asleep_reversed['date_time']
df_asleep_reversed['hr_ewm_span_60_from_onset'] = df_asleep_reversed.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(span=60, min_periods=180).mean())
df_asleep_reversed['hr_ewm_halflife_120_from_onset'] = df_asleep_reversed.groupby('date_sleep')['hr_clean'].transform(lambda x: x.ewm(halflife=120, min_periods=180).mean())

df_span_from_wake = df_asleep.groupby('date_sleep')['hr_ewm_span_60_from_wake'].apply(lambda x: x.tail(1)).reset_index()
df_halflife_from_wake = df_asleep.groupby('date_sleep')['hr_ewm_halflife_120_from_wake'].apply(lambda x: x.tail(1)).reset_index()
df_span_from_onset = df_asleep_reversed.groupby('date_sleep')['hr_ewm_span_60_from_onset'].apply(lambda x: x.tail(1)).reset_index()
df_halflife_from_onset = df_asleep_reversed.groupby('date_sleep')['hr_ewm_halflife_120_from_onset'].apply(lambda x: x.tail(1)).reset_index()

date_to_span_from_wake_dict = dict(zip(df_span_from_wake['date_sleep'], df_span_from_wake['hr_ewm_span_60_from_wake']))
date_to_halflife_from_wake_dict = dict(zip(df_halflife_from_wake['date_sleep'], df_halflife_from_wake['hr_ewm_halflife_120_from_wake']))
date_to_span_from_onset_dict = dict(zip(df_span_from_onset['date_sleep'], df_span_from_onset['hr_ewm_span_60_from_onset']))
date_to_halflife_from_onset_dict = dict(zip(df_halflife_from_onset['date_sleep'], df_halflife_from_onset['hr_ewm_halflife_120_from_onset']))

plt.hist(df_span_from_wake['hr_ewm_span_60_from_wake'].dropna(), alpha=.3)
plt.hist(df_halflife_from_wake['hr_ewm_halflife_120_from_wake'].dropna(), alpha=.3)
plt.hist(df_span_from_onset['hr_ewm_span_60_from_onset'].dropna(), alpha=.3)
plt.hist(df_halflife_from_onset['hr_ewm_halflife_120_from_onset'].dropna(), alpha=.3)

# save dict
with open('date_to_span_from_wake_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_span_from_wake_dict, picklefile)
with open('date_to_halflife_from_wake_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_halflife_from_wake_dict, picklefile)
with open('date_to_span_from_onset_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_span_from_onset_dict, picklefile)
with open('date_to_halflife_from_onset_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_halflife_from_onset_dict, picklefile)



# ----
# plot hr by time after sleep onset. cols are ea day.
# use actual time or time only asleep. start with asleep
# from the df above that created the ts that showed the 
# sleep stages. but then do with actual time, i.e., including
# time that fitbit says awake during sleep session. to see how different.
df_asleep.head()

sns.relplot(x='minutes_asleep', y='hr_interpolate_clean', 
            data=df_asleep,  # .sample(n=1000), 
            kind='line', ci=95, hue='alcohol_tertile', alpha=.5,
            palette=['green', 'orange', 'red'], 

df_asleep[['date_time', 'hr_clean', 'minutes_from_waking', 'minutes_asleep']]
df_asleep[df_asleep['hr_clean'].isnull()]

df_asleep_for_heatmap = df_asleep.copy(deep=True)
df_asleep_for_heatmap = df_asleep_for_heatmap.reset_index(drop=True)
df_asleep_for_heatmap = df_asleep_for_heatmap[['date_sleep', 'date_time', 'hr_clean', 
                                               'hr_interpolate_clean', 'minutes_asleep',
                                               'sleep_session']]

df_asleep_for_heatmap.groupby('date_sleep')['minutes_asleep'].max().max()

df_asleep_for_heatmap[df_asleep_for_heatmap['minutes_asleep']==16.083333333333332]
df_asleep_for_heatmap = df_asleep_for_heatmap[df_asleep_for_heatmap['date_sleep']!='2018-01-16']

df_asleep_for_heatmap[df_asleep_for_heatmap['minutes_asleep']==14.408333333333333]
df_asleep_for_heatmap = df_asleep_for_heatmap[df_asleep_for_heatmap['date_sleep']!='2017-12-23']

df_asleep_for_heatmap[df_asleep_for_heatmap['minutes_asleep']==13.716666666666667]
df_asleep_for_heatmap = df_asleep_for_heatmap[df_asleep_for_heatmap['date_sleep']!='2017-12-02']

df_asleep_for_heatmap['sleep_session'].value_counts()
df_asleep_for_heatmap = df_asleep_for_heatmap[df_asleep_for_heatmap['sleep_session']==1]

#max_hours = df_asleep_for_heatmap.groupby('date_sleep')['minutes_asleep'].max().max()
#date_sleep = df_asleep_for_heatmap[df_asleep_for_heatmap['minutes_asleep']==max_hours]['date_sleep'].values[0]
#df_date_sleep_base = df_asleep_for_heatmap[df_asleep_for_heatmap['date_sleep']==date_sleep]
#df_date_sleep_base = df_date_sleep_base.set_index('minutes_asleep')
#df_date_sleep_base = df_date_sleep_base[['hr_clean']]
#df_date_sleep_base.columns = ['hr_clean_base']

# create a df with min/hrs asleep df as index, thorugh 10 hours or something
x = df_asleep_for_heatmap['minutes_asleep'].unique()
x = np.sort(x)
df_date_sleep_base = pd.DataFrame(data=x, index=x)

for date_sleep in df_asleep_for_heatmap['date_sleep'].unique()[:]:
    df_date_sleep = df_asleep_for_heatmap[df_asleep_for_heatmap['date_sleep']==date_sleep]
    df_date_sleep = df_date_sleep.set_index('minutes_asleep')
    df_date_sleep = df_date_sleep[['hr_clean']]
    df_date_sleep.columns = ['hr_clean_'+str(date_sleep)[:10]]
    df_date_sleep_base = df_date_sleep_base.join(df_date_sleep)

del df_date_sleep_base[0]
df_date_sleep_base.index = np.round(df_date_sleep_base.index, 3)
# necessary? no.
#df_date_sleep_base.replace(np.nan, 0, inplace=True)

#df_date_sleep_base = df_date_sleep_base[df_date_sleep_base.index<9]

#cmap_enter = sns.diverging_palette(15, 125, sep=10, s=70, l=50, as_cmap=True)
cmap_enter = sns.diverging_palette(125, 15, sep=10, s=70, l=50, as_cmap=True)
cmap_enter="YlGnBu"

for col in df_date_sleep_base.columns:
    df_date_sleep_base[col] = df_date_sleep_base[col].interpolate(limit=1)

df_date_sleep_base.head()
df_date_sleep_base.tail()

fig = plt.figure(figsize=(10, 8))  
ax = sns.heatmap(df_date_sleep_base, cmap=cmap_enter)
plt.xticks([])
ax.xaxis.set_label_position("top") 
plt.xlabel('Days\n2017-05                             2018-01                             2018-09', 
           fontsize=16)
plt.ylabel('Hours from Sleep Onset', fontsize=16)
plt.ylim(1200,0)
#plt.yticks(fontsize=14)

df_asleep_for_heatmap['date_sleep'].min()
df_asleep_for_heatmap['date_sleep'].max()

# can i also create actual time from sleep onset (i.e., include the wake times)
# does this look any different?
# to do, resample by minute. now in real time, even if awake sometimes.
# then get minutes from sleep onset. do one keeping hr from awake. can see
# when i wake up and hr spikes. and one with these awake hr as null.
# can i plot the fitbit rem and deep over mine?
df_sleep_8_to_11_resampled.head()

# select just first sleep period and then only if greater than a certain amount of time?
# or i could groupby sleep sessions too? but plot all? seems ok?
# start by just selecting sleep sessoin 1.
df_sleep_8_to_11_resampled['sleep_session'].value_counts()
df_sleep_hr_for_heatmap = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['sleep_session']==1]
#df_sleep_hr_for_heatmap = df_sleep_8_to_11_resampled.copy(deep=True)

len(df_sleep_hr_for_heatmap.groupby('date_sleep').resample('30S', on='date_time').mean())  # 457006
len(df_sleep_hr_for_heatmap.groupby('date_sleep').resample('1min', on='date_time').mean())  # 228729

df_sleep_hr_for_heatmap = df_sleep_hr_for_heatmap.groupby('date_sleep').resample('1min', on='date_time').mean().reset_index()
df_sleep_hr_for_heatmap.head()
df_sleep_hr_for_heatmap['hr_clean']  # plenty of nulls. but should be ok.
df_sleep_hr_for_heatmap[df_sleep_hr_for_heatmap['awake']]
df_sleep_hr_for_heatmap['awake'].value_counts(normalize=True)
df_sleep_hr_for_heatmap[df_sleep_hr_for_heatmap['awake'].isnull()]

df_sleep_hr_for_heatmap.columns
df_sleep_hr_for_heatmap[['start_sleep', 'start_of_date_sleep']]  # the same?
# new col for each date sleep that has time of sleep onset
# then subtract from each time. should work?
df_first_minute_each_date_sleep = df_sleep_hr_for_heatmap[df_sleep_hr_for_heatmap['start_of_date_sleep']==1]
len(df_first_minute_each_date_sleep)
len(df_sleep_hr_for_heatmap['date_sleep'].unique())

date_sleep_to_first_time_dict = dict(zip(df_first_minute_each_date_sleep['date_sleep'], df_first_minute_each_date_sleep['date_time']))
df_sleep_hr_for_heatmap['time_of_sleep_onset'] = df_sleep_hr_for_heatmap['date_sleep'].map(date_sleep_to_first_time_dict)
df_sleep_hr_for_heatmap[['date_sleep', 'date_time', 'time_of_sleep_onset']]

df_sleep_hr_for_heatmap['hours_since_sleep_onset'] = (
        df_sleep_hr_for_heatmap['date_time'] - df_sleep_hr_for_heatmap['time_of_sleep_onset'] ) /  np.timedelta64(1, 'h')

df_sleep_hr_for_heatmap[['date_time', 'time_of_sleep_onset', 'hours_since_sleep_onset']]

df_sleep_hr_for_heatmap.groupby('date_sleep')['hours_since_sleep_onset'].max().hist(bins=15, alpha=.7)
plt.grid(False)
print(df_sleep_hr_for_heatmap['hours_since_sleep_onset'].mean())
# some nights with very little sleep. presume first of two sleep sessions?
# cut out after graphing.

df_sleep_hr_for_heatmap['hours_sleep'] = df_sleep_hr_for_heatmap.groupby('date_sleep')['hours_since_sleep_onset'].transform(lambda x: x.max())
df_sleep_hr_for_heatmap.groupby('date_sleep')['hours_sleep'].mean().hist(bins=15, alpha=.7)
plt.grid(False)

# remove days with fewer than 4 hours sleep
len(df_sleep_hr_for_heatmap['date_sleep'].unique())
df_sleep_hr_for_heatmap = df_sleep_hr_for_heatmap[df_sleep_hr_for_heatmap['hours_sleep']>=4]

df_sleep_hr_for_heatmap[df_sleep_hr_for_heatmap['date_sleep']=='2018-01-16']['hours_since_sleep_onset'].max()
# this had 16 hours since onset when look at both sleep sessions, i think

# functions to plot
def plot_hr_sleep_heatmap(df_sleep_hr_for_heatmap, metric):
    x = df_sleep_hr_for_heatmap['hours_since_sleep_onset'].unique()
    x = np.sort(x)
    df_date_sleep_base = pd.DataFrame(data=x, index=x)
    date_list = np.sort(df_sleep_hr_for_heatmap['date_sleep'].unique()[:])
    for date_sleep in date_list:
        df_date_sleep = df_sleep_hr_for_heatmap[df_sleep_hr_for_heatmap['date_sleep']==date_sleep]
        df_date_sleep = df_date_sleep.set_index('hours_since_sleep_onset')
        df_date_sleep = df_date_sleep[[metric]]
        df_date_sleep.columns = [metric+'_'+str(date_sleep)[:10]]
        df_date_sleep_base = df_date_sleep_base.join(df_date_sleep)

    del df_date_sleep_base[0]
    df_date_sleep_base.index = np.round(df_date_sleep_base.index, 3)
    return df_date_sleep_base

df_sleep_hr_for_heatmap['awake'].value_counts()
len(df_sleep_hr_for_heatmap[df_sleep_hr_for_heatmap['awake'].isnull()])

#df_sleep_hr_for_heatmap_no_awake = df_sleep_hr_for_heatmap[df_sleep_hr_for_heatmap['awake']==0]
df_date_sleep_base = plot_hr_sleep_heatmap(df_sleep_hr_for_heatmap, 'hr_clean')  # df_sleep_hr_for_heatmap  df_sleep_hr_for_heatmap_no_awake
df_date_sleep_base.head()

#len(df_date_sleep_base[df_date_sleep_base[df_date_sleep_base.columns].isnull()])
#for col in df_date_sleep_base.columns:
#    df_date_sleep_base[col] = df_date_sleep_base[col].interpolate(limit=1)
# doesn't actually change the number of nulls

df_sleep_hr_for_heatmap['hr_clean'].hist(bins=20, alpha=.7)
plt.grid(False)

def plot_heatmap_of_hr_over_each_night(df_date_sleep_base):
    cmap_enter = sns.diverging_palette(125, 15, sep=10, s=70, l=50, 
                                       center='light', as_cmap=True)
    cmap_enter="YlGnBu"  
    # this looks cooler. but green and red allows to see differnces better, allows to see increases in hr better
    # actually, not really true alltogether. the red-green obscures diffs toward the center-midpoint
    # the one shade is better for that
    df_date_sleep_base.head()
    df_date_sleep_base.tail()
    fig = plt.figure(figsize=(10, 8))  
    ax = sns.heatmap(df_date_sleep_base, cmap=cmap_enter, 
                     vmin=40, vmax=80)
    plt.xticks([])
    ax.xaxis.set_label_position("top") 
    plt.xlabel('Date\n2017-05                             2018-01                             2018-09', 
               fontsize=16)
    plt.ylabel('Hours from Sleep Onset', fontsize=16)
    plt.ylim(600,0)
    #plt.yticks(fontsize=14)

plot_heatmap_of_hr_over_each_night(df_date_sleep_base)

df_date_sleep_base.head()
# could at do heat map of rem to see if it's at the 1.4 hour mark

def plot_heatmap_of_rem_over_each_night(df_date_sleep_base):
    cmap_enter = sns.diverging_palette(125, 15, sep=10, s=70, l=50, 
                                       center='light', as_cmap=True)
    #cmap_enter="YlGnBu"  
    fig = plt.figure(figsize=(10, 8))  
    ax = sns.heatmap(df_date_sleep_base, cmap=cmap_enter, alpha=.25)
    plt.xticks([])
    ax.xaxis.set_label_position("top") 
    plt.xlabel('Date\n2017-05                             2018-01                             2018-09', 
               fontsize=16)
    plt.ylabel('Hours from Sleep Onset', fontsize=16)
    plt.ylim(600,0)
    #plt.yticks(fontsize=14)

df_date_sleep_base = plot_hr_sleep_heatmap(df_sleep_hr_for_heatmap, 'rem')  # df_sleep_hr_for_heatmap  df_sleep_hr_for_heatmap_no_awake
df_date_sleep_base = plot_hr_sleep_heatmap(df_sleep_hr_for_heatmap, 'deep')  # df_sleep_hr_for_heatmap  df_sleep_hr_for_heatmap_no_awake
df_date_sleep_base = plot_hr_sleep_heatmap(df_sleep_hr_for_heatmap, 'hr_clean')  # df_sleep_hr_for_heatmap  df_sleep_hr_for_heatmap_no_awake

plot_heatmap_of_rem_over_each_night(df_date_sleep_base)
# i guess it does roughly match up with the hr increases

# calculate hr increase for first rem, second rem, etc. and for deep.
# should really do it because rem really stands out first time. so 
# maybe that's especially telling?
def plot_heatmap_of_rem_for_one_night(df_date_sleep_base, date, metric):
    cmap_enter = sns.diverging_palette(125, 15, sep=10, s=70, l=50, 
                                       center='light', as_cmap=True)
    #cmap_enter="YlGnBu"  
    df_date_sleep_base_date = df_date_sleep_base[[metric+'_'+str(date)]]
    fig = plt.figure(figsize=(10, 8))  
    ax = sns.heatmap(df_date_sleep_base_date, cmap=cmap_enter, alpha=.25)
    plt.xticks([])
    ax.xaxis.set_label_position("top") 
    plt.ylabel('Hours from Sleep Onset', fontsize=16)
    #plt.ylim(600,0)

plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-11', 'rem')
plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-12', 'rem')
plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-13', 'rem')
plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-14', 'rem')
plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-17', 'rem')
plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-18', 'rem')
plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-19', 'rem')

plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-10', 'hr_clean')
plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-12', 'hr_clean')
plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-13', 'hr_clean')
plot_heatmap_of_rem_for_one_night(df_date_sleep_base, '2018-07-14', 'hr_clean')

# though if computed my own hr from about 1.25 hours to 2.25 hours, 
# i'd probably capture first rem. could compute max hr for 5 min here. 
# could also compute the increase in hr. maybe that's not meaingful? 
# could also get the max hr for 5 min for rem recorded by fitbit.
# actually, get rem and deep and light sleep using ewm from sleep onset
# and from waking, just like i did for entire night of hr. I did this above.
# so good for now.


# --------------
# activity data
# get the daily summary of activity. create date to steps, etc. dicts
df_activity_summary = pd.read_pickle('df_activity_summary.pkl')
df_activity_summary.shape
df_activity_summary.head()

df_activity_summary[['distance', 'steps']].corr()
# not exactly the same, but pretty much. 
# which means I should be able to cacluate
# pace? distance / active-time
# i.e., steps / time-with-steps-greater-than-zero

date_to_distance_dict = dict(zip(df_activity_summary['date'], df_activity_summary['distance']))
date_to_elevation_dict = dict(zip(df_activity_summary['date'], df_activity_summary['elevation']))
date_to_steps_dict = dict(zip(df_activity_summary['date'], df_activity_summary['steps']))
# save dict
with open('date_to_distance_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_distance_dict, picklefile)
with open('date_to_elevation_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_elevation_dict, picklefile)
with open('date_to_steps_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_steps_dict, picklefile)


# could just use this to try to get running. but about half
# of days don't have type of actiity. so don't know if should
# trust it
df_activity = pd.read_pickle('df_activity.pkl')
df_run = df_activity[df_activity['activity']=='Run']
df_run['date'] = pd.to_datetime(df_run['datetime'].dt.date)
df_run_time = df_run.groupby('date').size().reset_index().rename(columns={0:'minutes'})
df_run_time = df_run_time[df_run_time['minutes']<200]  # look at outlier on webpage
df_run_time['minutes'].hist(bins=12, alpha=.75)
plt.grid(False)
df_run[df_run['date']=='2018-04-19']
df_run[df_run['date']=='2018-04-20']

df_activity_summary[df_activity_summary['date']=='2018-04-18']
df_activity_summary[560:570]
# this is just wrong. maybe shouldnt trust this running measure. is there another?

# create function to get steps and elevation dicts. presume can sum and 
# will get same as daily count? but these give when i had stps and elevantino
# in case need timing of it. could get running here to if taking a lot of steps
# within a short amount of time. check those wonky dates -- april 19 and 20 -- 
# in this data here. at least just create a metric that's the number of steps
# over time or something. to see if total steps is all that matters or if get
# more bang for the buck if it's done in a shorter time interval. i suppose it's
# basically getting the number of minutes with at least one step is the second
# metric here that i can measure and maybe combine.

# get floors instead of elevation. it's the same (elevation is floors * 10)
# and i have floors data downloaded. don't have elevation.

# steps

date = date_list[0]
metric = 'floors'


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


activity_data_day_dict.keys()

#date_list = pd.date_range('2016-10-01', '2018-09-25', freq='D')
#date = date_list[100]
#date = date_list[0]
#
#steps_data_day_dict = open_metric_dict(date, 'steps')
#steps_data_day_dict = open_metric_dict(date, 'elevation')
#
#date_list = pd.date_range('2017-07-25', '2017-07-25', freq='D')
#date = dates[100]
#date = date_list[0]
#
#floors_data_day_dict = open_metric_dict(date_list[0], 'floors')
#
#
#steps_data_day_dict.keys()
#steps_data_day_dict['activities-steps']  # give the total number of steps within the day
#steps_data_day_dict['activities-steps-intraday'].values()
#steps_data_day_dict['activities-steps-intraday'].keys()
#steps_data_day_dict['activities-steps-intraday']['dataset'][1050]
#
## these two just say the dataset is in 1-min increments
#steps_data_day_dict['activities-steps-intraday']['datasetInterval']
#steps_data_day_dict['activities-steps-intraday']['datasetType']
#
#
#steps_data_day_dict.keys()


def create_df_for_metric_each_minute(date_list, metric):
    time_and_metric_dict = {'date':[], 'time':[], metric:[]}
    for date in date_list:
        metric_data_day_dict = open_metric_dict(date, metric)
        if metric_data_day_dict == None:
            None
        else:
            dataset_day = metric_data_day_dict['activities-'+metric+'-intraday']['dataset']
            for data_dict in dataset_day:
                time_and_metric_dict['date'].append(date)
                time_and_metric_dict['time'].append(data_dict['time'])
                time_and_metric_dict[metric].append(data_dict['value'])
    df_time_metric = pd.DataFrame(time_and_metric_dict)
    return df_time_metric  
 
#date_list = date_list[100:105]
#date_list = pd.date_range('2017-10-01', '2017-10-05', freq='D')
date_list = pd.date_range('2016-10-01', '2018-09-25', freq='D')
df_time_steps = create_df_for_metric_each_minute(date_list, 'steps')

df_time_steps.groupby('date')['steps'].sum().hist(bins=15, alpha=.6)
plt.grid(False)

# active minutes
df_time_steps.groupby('date')['steps'].apply(lambda x: len(x[x>0])).hist(bins=15, alpha=.6)
plt.grid(False)

date_to_steps_dict = dict(df_time_steps.groupby('date')['steps'].sum())
date_to_active_minutes_dict = dict(df_time_steps.groupby('date')['steps'].apply(lambda x: len(x[x>0])))

# save dicts
with open('date_to_steps_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_steps_dict, picklefile)
with open('date_to_active_minutes_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_active_minutes_dict, picklefile)


# create some sort of metric that gets at intensity of steps
# get at overall steps per minute that took at least one step.
# but come up with something additional to that, that gets at
# peak intensity in some way? how? e.g., simple would be to take
# top 10 minutes with the most steps and avarege those. but is
# there an even more nuanced way?
df_time_steps[df_time_steps['steps']>0]['steps'].hist(bins=25, alpha=.7)
plt.grid(False)
# this is really cool! bi-modal. assuming this is walking vs. running    
for date in df_time_steps['date'].unique()[100:105]:
    df_date_steps = df_time_steps[df_time_steps['date']==date]
    df_date_steps[df_date_steps['steps']>0]['steps'].hist(bins=10, 
                 color='red', alpha=.3)
    plt.grid(False)
    plt.xlim(0,150)
    plt.ylim(0,60)
    plt.show();

# how to calculate how intense my most intense walking-running was that day?
# i want to weight the intense stuff more, and the most intense, weight tbe most?
# order minutes by number of steps and get ewm wtih drop-off after 30 min? 15 min?
# conceptually, it's how intense was the most intense steps on that day.
df_time_steps = df_time_steps.sort_values(by=['date', 'steps'])

# create several intensity metrics -- with diff spands of time before starts weighting less
df_time_steps['steps_intensity_15_min'] = df_time_steps.groupby('date')['steps'].transform(lambda x: x.ewm(span=15).mean())  
df_time_steps['steps_intensity_30_min'] = df_time_steps.groupby('date')['steps'].transform(lambda x: x.ewm(span=30).mean())  
df_time_steps['steps_intensity_60_min'] = df_time_steps.groupby('date')['steps'].transform(lambda x: x.ewm(span=60).mean())  

df_time_steps[['steps_intensity_15_min', 'steps']]
df_time_steps[['steps_intensity_15_min', 'steps_intensity_60_min']]

df_time_steps_intensity_15_min = df_time_steps.groupby('date')['steps_intensity_15_min'].apply(lambda x: x.tail(1)).reset_index()
df_time_steps_intensity_30_min = df_time_steps.groupby('date')['steps_intensity_30_min'].apply(lambda x: x.tail(1)).reset_index()
df_time_steps_intensity_60_min = df_time_steps.groupby('date')['steps_intensity_60_min'].apply(lambda x: x.tail(1)).reset_index()

df_time_steps_intensity_15_min['steps_intensity_15_min'].hist(alpha=.7, bins=25)
plt.grid(False)
# are these super low dates outliers, bad values, or real?
df_time_steps_intensity_60_min['steps_intensity_60_min'].hist(alpha=.7, bins=25)
plt.grid(False)

date_to_steps_intensity_15_min_dict = dict(zip(df_time_steps_intensity_15_min['date'], df_time_steps_intensity_15_min['steps_intensity_15_min']))
date_to_steps_intensity_30_min_dict = dict(zip(df_time_steps_intensity_30_min['date'], df_time_steps_intensity_30_min['steps_intensity_30_min']))
date_to_steps_intensity_60_min_dict = dict(zip(df_time_steps_intensity_60_min['date'], df_time_steps_intensity_60_min['steps_intensity_60_min']))

# save dicts
with open('date_to_steps_intensity_15_min_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_steps_intensity_15_min_dict, picklefile)
with open('date_to_steps_intensity_30_min_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_steps_intensity_30_min_dict, picklefile)
with open('date_to_steps_intensity_60_min_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_steps_intensity_60_min_dict, picklefile)

# elevation
#date_list = date_list[100:105]
date_list = pd.date_range('2016-10-01', '2018-09-25', freq='D')
df_time_floors = create_df_for_metric_each_minute(date_list, 'floors')

df_time_floors.groupby('date')['floors'].sum().hist(bins=50, alpha=.6)
plt.grid(False)

df_floors_by_date = df_time_floors.groupby('date')['floors'].sum()
plt.plot(df_floors_by_date)
df_floors_by_date[70:225]

# all zero floors. try to get again. or get elevation instead?
# checked again and the api for minute-by-minute activity data is
# just producing zero steps/elvation for this data rate. so need
# to use summary data. but wasn't planning to do anything yet
# related to timing of the steps (though in future maybe could 
# match this up with hr to get a sense for how strenous going up was?)
date_list = pd.date_range('2016-12-12', '2017-05-07', freq='D')

df_activity_elevation = df_activity_summary.set_index('date')
df_activity_elevation['elevation'] = df_activity_elevation['elevation']/10
plt.plot(df_activity_elevation['elevation'])

# just use this elevation summary for now, for floors
# rather than summing minute by minute floors data
df_activity_elevation = df_activity_elevation.reset_index()
df_activity_elevation.dtypes
date_to_floors_dict = dict(zip(df_activity_elevation['date'], df_activity_elevation['elevation']))

# save dicts
with open('date_to_floors_dict.pkl', 'wb') as picklefile:
	pickle.dump(date_to_floors_dict, picklefile)

# same in json file downloaded from website.
df_activity_elevation[df_activity_elevation['date']=='2017-05-16']










# ------------------------
# create past lagged vars
def create_lagged_past_date_variables(number_of_lags, date_variable, df):
    for lag in list(range(1,number_of_lags+1)):
        df['date_past_'+str(lag)] = df[date_variable] - timedelta(days=lag)
    return df

df_hr_mean = create_lagged_past_date_variables(1, 'date_sleep', df_hr_mean)
df_hr_mean[['date_sleep', 'date_past_1']]

df_hr_mean = create_lagged_past_date_variables(2, 'date_sleep', df_hr_mean)
df_hr_mean[['date_sleep', 'date_past_1', 'date_past_2']]

df_hr_mean['minutes_sleep_lag_1'] = df_hr_mean['date_past_1'].map(date_to_minutes_alseep_dict)
df_hr_mean[['min_asleep', 'minutes_sleep_lag_1']]

results = smf.ols(formula = """hr_rest_next_day ~ hr_mean + hr_mean_lag1 + 
                  hr_rest + I(hr_rest**2) + min_asleep + minutes_sleep_lag_1 + 
                  temp + month_ts + start_sleep_time + tue + wed + thu + fri + 
                  sat + sun""",  #  + alcohol + alcohol_lag1 
                  data=df_hr_mean).fit()
print(results.summary())
# the more minutes asleep, the lower resting hr next day

df_hr_mean[['hr_mean', 'min_asleep']].corr()
sns.lmplot(x='min_asleep', y='hr_mean', data=df_hr_mean[df_hr_mean['min_asleep']<800], 
           scatter_kws={'alpha':.15}, order=3)
plt.xlim(250, 700)
plt.ylim(46, 65)

results = smf.ols(formula = """hr_rest_next_day ~ hr_mean + hr_mean_lag1 + 
                  hr_rest + I(hr_rest**2) + min_asleep + minutes_sleep_lag_1 + 
                  temp + month_ts + start_sleep_time + tue + wed + thu + fri + 
                  sat + sun + hr_mean:min_asleep""",  #  + alcohol + alcohol_lag1 
                  data=df_hr_mean).fit()
print(results.summary())
# no interaction between min asleep and hr while asleep

df_hr_mean['minutes_within_sleep'] = df_hr_mean['date_sleep'].map(date_to_min_within_sleep_dict)
df_hr_mean['minutes_within_sleep_lag1'] = df_hr_mean['date_past_1'].map(date_to_min_within_sleep_dict)
df_hr_mean[['minutes_within_sleep', 'minutes_within_sleep_lag1']]
df_hr_mean['sleep_efficiency'] = df_hr_mean['min_asleep'] / df_hr_mean['minutes_within_sleep'] 
df_hr_mean['sleep_efficiency'].hist()
df_hr_mean[df_hr_mean['sleep_efficiency']==1]
# look up these days. can't be 100% efficient!
# something is wrong, need to do over.
# it'd be nice to get the number of sleep interruptions too (vs. time awake during sleep session)

results = smf.ols(formula = """hr_rest_next_day ~ hr_mean + hr_mean_lag1 + 
                  hr_rest + I(hr_rest**2) + min_asleep + minutes_sleep_lag_1 + 
                  temp + month_ts + start_sleep_time + tue + wed + thu + fri + 
                  sat + sun + sleep_efficiency""",  #  + alcohol + alcohol_lag1 
                  data=df_hr_mean[df_hr_mean['sleep_efficiency']!=1]).fit()
print(results.summary())
# sleep_efficiency not adding anything to story

# look at changes in hr instead of this lagged var approach
df_hr_mean[['sleep_efficiency', 'hr_mean', 'min_asleep']].corr()
sns.lmplot(x='hr_mean', y='sleep_efficiency', data=df_hr_mean, 
           scatter_kws={'alpha':.15}, order=2)

# get subjectve ratings for day after sleep
df_hr_mean.columns
date_to_energy_dict = dict(zip(df_hr_mean['date_sleep'], df_hr_mean['energy']))
date_to_fun_dict = dict(zip(df_hr_mean['date_sleep'], df_hr_mean['fun']))
date_to_subj_sleep_dict = dict(zip(df_hr_mean['date_sleep'], df_hr_mean['subj_sleep']))

df_hr_mean['energy_next_day'] = df_hr_mean['date_forward_1'].map(date_to_energy_dict)
df_hr_mean['fun_next_day'] = df_hr_mean['date_forward_1'].map(date_to_fun_dict)
df_hr_mean['subj_sleep_next_day'] = df_hr_mean['date_forward_1'].map(date_to_subj_sleep_dict)

sns.countplot(df_hr_mean['subj_sleep_next_day'])

# should add another lag to min sleep and to hr rest
df_hr_mean['hr_rest_lag1'] = df_hr_mean['date_past_1'].map(date_to_hr_resting_dict)
df_hr_mean['minutes_sleep_lag_2'] = df_hr_mean['date_past_2'].map(date_to_minutes_alseep_dict)
df_hr_mean[['date_sleep', 'min_asleep', 'minutes_sleep_lag_1' ,'minutes_sleep_lag_2']]

results = smf.ols(formula = """energy_next_day ~ hr_mean + hr_mean_lag1 + 
                  hr_rest + hr_rest_lag1 + min_asleep + minutes_sleep_lag_1 + minutes_sleep_lag_2 +
                  temp + month_ts + start_sleep_time + tue + wed + thu + fri + 
                  sat + sun + energy + energy_lag1 + fun + fun_lag1 + 
                  subj_sleep""",  #  + alcohol + alcohol_lag1 
                  data=df_hr_mean).fit()
print(results.summary())
# more min asleep, less energy. guess this is plausible. and min asleep two
# nights ago is the opposite. more min two nights ago = more energy!

results = smf.ols(formula = """fun_next_day ~ hr_mean + hr_mean_lag1 + 
                  hr_rest + hr_rest_lag1 + min_asleep + minutes_sleep_lag_1 + minutes_sleep_lag_2 + 
                  temp + month_ts + start_sleep_time + tue + wed + thu + fri + 
                  sat + sun + energy + energy_lag1 + fun + fun_lag1 + 
                  subj_sleep""",  #  + alcohol + alcohol_lag1 
                  data=df_hr_mean).fit()
print(results.summary())

results = smf.ols(formula = """subj_sleep_next_day ~ hr_mean + hr_mean_lag1 + 
                  hr_rest + hr_rest_lag1 + min_asleep + minutes_sleep_lag_1 + minutes_sleep_lag_2 + 
                  temp + month_ts + start_sleep_time + tue + wed + thu + fri + 
                  sat + sun + energy + energy_lag1 + fun + fun_lag1 + 
                  subj_sleep""",  #  + alcohol + alcohol_lag1 
                  data=df_hr_mean).fit()
print(results.summary())


df_hr_mean[['energy_next_day', 'fun_next_day', 
            'subj_sleep_next_day', 'hr_mean',
            'hr_rest_next_day', 'hr_rest_min_next_day']].corr()

# sleep min X hr?
results = smf.ols(formula = """hr_rest_min_next_day ~ hr_mean + hr_mean_lag1 + 
                  hr_rest_min + min_asleep + minutes_sleep_lag_1 + temp + 
                  month_ts + start_sleep_time + tue + wed + thu + fri + 
                  sat + sun + energy  + fun + subj_sleep + hr_mean*hr_rest_min""",  #  + alcohol + alcohol_lag1 
                  data=df_hr_mean).fit()
print(results.summary())



# LEFT OFF. WHAT NEXT? SEE IDEAS IN 2/9 NOTEBOOK.
# GET EXERCISE. THEN MAKE NEW SLEEP AND OTHER METRICS.
















# ==================================================================
# tigramite
# examine causal network of fitness and sleep metrics from fitbit with trigramite
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction


def plot_ts(df_activity_sleep, variable_list, i):
    plt.figure(figsize=(8,5))
    plt.plot(df_activity_sleep['date_sleep'], df_activity_sleep[variable_list[i]], 
             alpha=.6, color='green', linewidth=1.5)
    plt.xticks(fontsize=14)  # rotation=30
    plt.yticks(fontsize=14)
    plt.xlabel('')
    plt.ylabel(variable_list[i], fontsize=18);
    plt.title(variable_list[i], fontsize=18)
    sns.despine()
    plt.show();

df_hr_mean.columns

df_hr_mean['hr_rest_lag1'] = df_hr_mean['hr_rest'].shift(1)
df_hr_mean[['hr_rest', 'hr_rest_lag1']]

# with tigramite, this only gets at efx of sleep on next day stuff
# would have to change the lags to get at day-stuff on sleep efx.
variable_list = ['hr_mean', 'start_sleep_time', 'min_asleep', 'subj_sleep',
                 'sleep_efficiency', 'temp', 'energy_lag1', 'fun_lag1', 
                 'hr_rest_lag1']

df_hr_mean[variable_list].corr()
df_hr_mean[variable_list+['minutes_sleep_lag_1']].corr()


for i in range(1, len(variable_list)):
    plot_ts(df_hr_mean, variable_list, i)


df_for_tigramite = df_hr_mean.copy(deep=True)

# fill nans with mean up to that point in time
for col in variable_list:
    print(col)
    df_for_tigramite[col+'_expanding'] = df_for_tigramite[col].shift(1).expanding(min_periods=4).mean()
    df_for_tigramite.loc[df_for_tigramite[col].isnull(), col] = df_for_tigramite[col+'_expanding']

# check for missing values now
for col in variable_list:
    print(col, len(df_for_tigramite[df_for_tigramite[col].isnull()]))

df_for_tigramite = df_for_tigramite.dropna()

# standardiz/z variables
for col in variable_list:
    print(col)
    df_for_tigramite[col] = (df_for_tigramite[col] - df_for_tigramite[col].mean()) / df_for_tigramite[col].std()

df_for_tigramite[variable_list].mean()
df_for_tigramite[variable_list].std()


data = df_for_tigramite[variable_list].values
T, N = data.shape

# Initialize dataframe object
dataframe = pp.DataFrame(data)

# Specify time axis and variable names
datatime = np.arange(len(data))
var_names = variable_list

tp.plot_timeseries(data, datatime, var_names)

parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=parcorr,
    var_names=var_names,
    verbosity=1)

correlations = pcmci.get_lagged_dependencies(tau_max=3)
lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names':var_names, 
                                    'x_base':5, 'y_base':.5})

# start wtih tau_max=3 for now. look at autocorr plots to think more
pcmci.verbosity = 1
results = pcmci.run_pcmci(tau_max=3, pc_alpha=None)

print("p-values")
print (results['p_matrix'].round(3))
print("MCI partial correlations")
print (results['val_matrix'].round(2))

q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'])  # , fdr_method='fdr_bh'
pcmci._print_significant_links(
        p_matrix = results['p_matrix'], 
        q_matrix = results['p_matrix'],  # q_matrix  results['p_matrix']
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)

link_matrix = pcmci._return_significant_parents(pq_matrix=results['p_matrix'],  # q_matrix, results['p_matrix']
                        val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']


# ==========================================================
# pickle for jupyter nb viz
cd /Users/charlesmartens/Google Drive/jobs/data science fellowships/

with open('results_my_fitbit.pkl', 'wb') as picklefile:
	pickle.dump(results, picklefile)
#with open('results_my_fitbit.pkl', 'rb') as picklefile:
#	results_my_fitbit = pickle.load(picklefile)

with open('link_matrix_my_fitbit.pkl', 'wb') as picklefile:
	pickle.dump(link_matrix, picklefile)
#with open('link_matrix_my_fitbit.pkl', 'rb') as picklefile:
#	link_matrix_my_fitbit = pickle.load(picklefile)

with open('var_names_my_fitbit.pkl', 'wb') as picklefile:
	pickle.dump(var_names, picklefile)
#with open('var_names_my_fitbit.pkl', 'rb') as picklefile:
#	var_names_my_fitbit = pickle.load(picklefile)
# ==========================================================


tp.plot_graph(
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=var_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    )

# Plot time series graph
tp.plot_time_series_graph(
    val_matrix=results['val_matrix'],
    link_matrix=link_matrix,
    var_names=var_names,
    link_colorbar_label='MCI',
    )

# ==================================================================


# think more about predicting resting hr and energy based on prior 
# levels. i don't want to penelaze days where the prior day's metric
# was already good and so there's little room for movement to get 
# even better. maybe if i diff the dv and control for prior level?
# or do that pct rank of where on distribution from the regression at
# that particular level the value of the dv falls. not explaining it 
# well here.

# can/should get hr at diff times in sleep cycle and early vs late in night
# to distinguish/predict metrics next day. can think about getting low/min
# hr while asleep score each night. can look at fluctuations in hr (std 
# or something similar) or mean x std interaction.

# compute cumulative variables, i.e., weighted avg of past x days of sleep minutes and sleep hr?
# diff the ts to examine that way. to de-trend or stationarize the ts

# try differencing approach or logged differencing
df_hr_mean.shape
df_hr_mean[['hr_rest_next_day', 'hr_rest']]

def create_log_diff_measure(df_hr_mean, variable_first, variable_second):
    df_hr_mean[variable_first+'_diff'] = df_hr_mean[variable_second] - df_hr_mean[variable_first] 
    df_hr_mean[variable_first+'_log_diff'] = np.log(df_hr_mean[variable_second]) - np.log(df_hr_mean[variable_first])
    df_hr_mean[variable_first+'_diff'].hist(alpha=.6, bins=15)
    plt.show();    
    df_hr_mean[variable_first+'_log_diff'].hist(alpha=.6, bins=15)
    plt.show();    
    return df_hr_mean

df_hr_mean = create_log_diff_measure(df_hr_mean, 'hr_rest', 'hr_rest_next_day')
df_hr_mean.columns

variable = 'hr_rest_diff'  # 'hr_rest_next_day' 'hr_rest_diff' 'hr_rest_log_diff' 
plt.plot(df_hr_mean['date_sleep'], df_hr_mean[variable])
plt.xticks(rotation=30)

df_hr_mean = create_log_diff_measure(df_hr_mean, 'hr_mean_lag1', 'hr_mean')
variable = 'hr_mean_lag1_log_diff'  # 'hr_mean_lag1_log_diff' 'hr_mean_lag1_diff'  'hr_mean_lag1'
plt.plot(df_hr_mean['date_sleep'], df_hr_mean[variable])
plt.xticks(rotation=30)

df_hr_mean = create_log_diff_measure(df_hr_mean, 'minutes_sleep_lag_1', 'min_asleep')
variable = 'minutes_sleep_lag_1_log_diff'  # 'hr_mean_lag1_log_diff' 'hr_mean_lag1_diff'  'hr_mean_lag1'
plt.plot(df_hr_mean['date_sleep'], df_hr_mean[variable])
plt.xticks(rotation=30)

df_hr_mean = create_log_diff_measure(df_hr_mean, 'fun', 'fun_next_day')
variable = 'fun_log_diff'  # 'hr_mean_lag1_log_diff' 'hr_mean_lag1_diff'  'hr_mean_lag1'
plt.plot(df_hr_mean['date_sleep'], df_hr_mean[variable])
plt.xticks(rotation=30)


results = smf.ols(formula = """hr_rest_next_day ~ hr_mean + hr_mean_lag1 + 
                  hr_rest + min_asleep + minutes_sleep_lag_1 + temp + 
                  month_ts + start_sleep_time + tue + wed + thu + fri + 
                  sat + sun""",  #  + alcohol + alcohol_lag1 + sleep_efficiency
                  data=df_hr_mean).fit()
print(results.summary())

results = smf.ols(formula = """hr_rest_log_diff ~ hr_mean_lag1_log_diff + 
                  minutes_sleep_lag_1_log_diff + hr_rest + I(hr_rest**2) + temp + 
                  tue + wed + thu + fri + sat + sun + hr_mean_lag1 + minutes_sleep_lag_1 
                  """,  #  + alcohol + alcohol_lag1 + sleep_efficiency
                  data=df_hr_mean).fit()
print(results.summary())
# + hr_mean_lag1:hr_mean_lag1_log_diff + minutes_sleep_lag_1:minutes_sleep_lag_1_log_diff

sns.lmplot(x='hr_mean_lag1_log_diff', y='hr_rest_diff', data=df_hr_mean, x_bins=5)
sns.lmplot(x='minutes_sleep_lag_1_log_diff', y='hr_rest_diff', data=df_hr_mean, x_bins=5)


results = smf.ols(formula = """fun_log_diff ~ hr_mean_lag1_log_diff + 
                  minutes_sleep_lag_1_log_diff + hr_rest + I(hr_rest**2) + temp + 
                  tue + wed + thu + fri + sat + sun + hr_mean_lag1 + 
                  minutes_sleep_lag_1 + fun + I(fun**2) + I(fun**3)
                  + hr_mean_lag1_log_diff:hr_mean 
                  + minutes_sleep_lag_1_log_diff:min_asleep""",  #  + alcohol + alcohol_lag1 + sleep_efficiency
                  data=df_hr_mean).fit()
print(results.summary())



# ---------------
# explore statsmodels ts package





# ---------------
# explore alcohol with other sleep metrics:
# time, wakings, std, ? (freq band of sleep cycles? - wait)
# row = night's sleep
df_sleep_8_to_11_resampled.head()
df_sleep_8_to_11_resampled.groupby

df_sleep_8_to_11.head()
df_sleep_8_to_11['deep'].unique()
df_sleep_8_to_11['light'].unique()
df_sleep_8_to_11['rem'].unique()
df_sleep_8_to_11['restless'].unique()
df_sleep_8_to_11['awake'].unique()

df_rows_asleep = df_sleep_8_to_11[df_sleep_8_to_11['awake']!=1].groupby('date_sleep').size().reset_index().rename(columns={0:'alseep_30sec'})
df_rows_asleep['alseep_30sec'].hist(bins=20)
df_rows_asleep['alseep_30sec'].mean() / 120
df_rows_asleep['alseep_30sec'].median() / 120

sns.relplot(x='date_sleep', y='alseep_30sec', data=df_rows_asleep, kind='line')
df_rows_asleep = df_rows_asleep.sort_values(by='date_sleep')

df_rows_asleep.set_index('date_sleep', inplace=True)
df_rows_asleep['time_asleep_rolling'] = df_rows_asleep['alseep_30sec'].rolling(window='60D', min_periods=10).mean()
df_rows_asleep['date_sleep'] = df_rows_asleep.index
sns.relplot(x='date_sleep', y='time_asleep_rolling', data=df_rows_asleep, kind='line')
plt.xticks(rotation=30)
sns.relplot(x='date_sleep', y='alseep_30sec', data=df_rows_asleep, kind='line')
plt.xticks(rotation=30)

# delete this one outlier night?
sorted(df_rows_asleep['alseep_30sec'].values)



df_rows_asleep['alcohol'] = df_rows_asleep['date_sleep'].map(date_to_alcohol_dict)
len(df_rows_asleep[df_rows_asleep['alcohol'].notnull()])

df_rows_asleep['alcohol_rolling'] = df_rows_asleep['alcohol'].rolling(window='60D').mean()
sns.relplot(x='date_sleep', y='alcohol_rolling', data=df_rows_asleep, kind='line')

sns.countplot(x='alcohol', data=df_rows_asleep, alpha=.7, color='dodgerblue')
df_rows_asleep['alcohol'].replace([4,5,6],[3,3,3], inplace=True)

sns.barplot(x='alcohol', y='alseep_30sec', data=df_rows_asleep, 
            color='dodgerblue', errcolor='grey', alpha=.6)
# but should probably be controlling for trend over time
# same with hr mean? is there a linear trend over time for hr mean to drop? maybe not
# but could also control for hr the day or week before the alcohol. better design?
# i.e., alcohol related to hr that night controlling for how hr has been lately


df_rows_restless = df_sleep_8_to_11.groupby('date_sleep')['restless'].sum().reset_index()
df_rows_deep = df_sleep_8_to_11.groupby('date_sleep')['deep'].sum().reset_index().rename(columns={0:'deep_30sec'})
df_rows_rem = df_sleep_8_to_11.groupby('date_sleep')['rem'].sum().reset_index().rename(columns={0:'rem_30sec'})

df_rows_restless['restless'].hist()
df_rows_restless = df_rows_restless[df_rows_restless['restless']>0]
df_rows_restless.set_index('date_sleep', inplace=True)
df_rows_restless['time_restless_rolling'] = df_rows_restless['restless'].rolling(window='60D').mean()
df_rows_restless['date_sleep'] = df_rows_restless.index
sns.relplot(x='date_sleep', y='time_restless_rolling', data=df_rows_restless, kind='line')

df_rows_restless['alcohol'] = df_rows_restless['date_sleep'].map(date_to_alcohol_dict)
len(df_rows_restless[df_rows_restless['alcohol'].notnull()])

sns.countplot(x='alcohol', data=df_rows_restless, alpha=.7, color='dodgerblue')
df_rows_restless['alcohol'].replace([4,5,6],[3,3,3], inplace=True)

sns.barplot(x='alcohol', y='restless', data=df_rows_restless, 
            color='dodgerblue', errcolor='grey', alpha=.6)


def explore_type_of_sleep(df_rows_restless, date_to_alcohol_dict, variable):
    print(df_rows_restless[variable].hist(bins=20, alpha=.6))
    df_rows_restless = df_rows_restless[df_rows_restless[variable]>0]
    print(df_rows_restless[variable].hist(bins=20, alpha=.6))
    df_rows_restless.set_index('date_sleep', inplace=True)
    df_rows_restless['time_'+variable+'_rolling'] = df_rows_restless[variable].rolling(window='60D').mean()
    df_rows_restless['date_sleep'] = df_rows_restless.index
    sns.relplot(x='date_sleep', y='time_'+variable+'_rolling', data=df_rows_restless, kind='line')
    plt.show(); 
    df_rows_restless['alcohol'] = df_rows_restless['date_sleep'].map(date_to_alcohol_dict)
    print('days with alcohol rating', len(df_rows_restless[df_rows_restless['alcohol'].notnull()])) 
    print(sns.countplot(x='alcohol', data=df_rows_restless, alpha=.7, color='dodgerblue'))
    plt.show();
    df_rows_restless['alcohol'].replace([4,5,6],[3,3,3], inplace=True)
    print(sns.countplot(x='alcohol', data=df_rows_restless, alpha=.7, color='dodgerblue'))
    plt.show();
    sns.barplot(x='alcohol', y=variable, data=df_rows_restless, 
                color='dodgerblue', errcolor='grey', alpha=.6)
    return df_rows_restless

df_rows_restless = explore_type_of_sleep(df_rows_restless, date_to_alcohol_dict, 'restless')
df_rows_deep = explore_type_of_sleep(df_rows_deep, date_to_alcohol_dict, 'deep')
df_rows_rem = explore_type_of_sleep(df_rows_rem, date_to_alcohol_dict, 'rem')


# add in subjective sleep quality, fun, and energy ratings
# get date the next day. because want to see hr at night and ratings the next day
df_daily_qs['date_next'] = df_daily_qs['date'] + timedelta(days=1)
df_daily_qs[['date_next', 'date']]
df_daily_qs.columns

# NO PATTERNS BELOW! AM I MAPPING TO THE CORRECT DAY??? AH, MAYBE I SHOULD DO PRIOR DAY
# BECAUSE THAT'S THE NIGHT'S SLEEP I WANT TO MATCH TO
date_to_subjective_sleep_dict = dict(zip(df_daily_qs['date_prior'], df_daily_qs['sleep']))
date_to_fun_dict = dict(zip(df_daily_qs['date_prior'], df_daily_qs['fun']))
date_to_energy_dict = dict(zip(df_daily_qs['date_prior'], df_daily_qs['energy']))

df_hr_mean['subj_sleep'] = df_hr_mean['date_sleep'].map(date_to_subjective_sleep_dict)
len(df_hr_mean[df_hr_mean['subj_sleep'].notnull()])

sns.countplot(x='subj_sleep', data=df_hr_mean, alpha=.7, color='dodgerblue')

sns.barplot(x='subj_sleep', y='hr_mean', data=df_hr_mean, alpha=.6, 
            color='dodgerblue', errcolor='grey')
plt.ylim(40,65)
plt.ylabel('Mean Heart Rate', fontsize=15)
plt.xlabel('Subjective Sleep Quality', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()


df_hr_mean['energy'] = df_hr_mean['date_sleep'].map(date_to_energy_dict)
len(df_hr_mean[df_hr_mean['energy'].notnull()])

sns.countplot(x='energy', data=df_hr_mean, alpha=.7, color='dodgerblue')
df_hr_mean['energy'].value_counts()
df_hr_mean['energy'].replace([0,-1], [np.nan, np.nan], inplace=True)

sns.barplot(x='energy', y='hr_mean', data=df_hr_mean, alpha=.6, 
            color='dodgerblue', errcolor='grey')
plt.ylim(40,65)
plt.ylabel('Mean Heart Rate', fontsize=15)
plt.xlabel('Energy', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()


df_hr_mean['fun'] = df_hr_mean['date_sleep'].map(date_to_fun_dict)
len(df_hr_mean[df_hr_mean['fun'].notnull()])

sns.countplot(x='fun', data=df_hr_mean, alpha=.7, color='dodgerblue')
df_hr_mean['fun'].value_counts()

sns.barplot(x='fun', y='hr_mean', data=df_hr_mean, alpha=.6, 
            color='dodgerblue', errcolor='grey')
plt.ylim(40,65)
plt.ylabel('Mean Heart Rate', fontsize=15)
plt.xlabel('Fun', fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.despine()

df_hr_mean.reset_index(inplace=True)

df_hr_mean[['energy', 'subj_sleep', 'fun', 'hr_mean', 'index']].corr()


results = smf.ols(formula = 'hr_mean ~ energy', data=df_hr_mean).fit()
print(results.summary())
results.pvalues.energy

results = smf.ols(formula = 'hr_mean ~ fun + index', data=df_hr_mean).fit()
print(results.summary())
results.pvalues.fun

results = smf.ols(formula = 'hr_mean ~ subj_sleep + index', data=df_hr_mean).fit()
print(results.summary())
results.pvalues.subj_sleep

# also try controlling for the prior day's x. makes sense i think
# e.g., relation between hr-sleep and fun, controlling for yesteray's fun
# i.e., does hr last night lead to higher fun that what we'd expect given
# fun the day before? or could do fun the week before -- e.g., 7-day 
# moving avg. seems to make sense. then might not have to control for index,
# which seems could be weird to do. but does it matter that this is within
# subj anys? i.e., when comparing diff fun ratings, it's always my own, 
# it's always relative to my own fun ratings. do i think it's about
# an increase in fun, though? if hr is low, should there be an increase
# in fun? what if fun was already high? this could be a time to do that
# cubic thing -- enter hr and fun cubed to predict fun the next day.
# or x between hr and fun prior day to predict fun the next day

# do i need to worry about hr prior? it's already the predictor. but if
# i also include hr from two days ago, or past week, it's saying, does
# hr from last night go above and beyond hr recently to predict fun
# or change in fun.

# corr matrix -- how are things all correlated?
# does hr at night predict stuff the next day better than anything else?

# hr two nights out -- maybe two nights ahead and two nights before alcohol

















# -----------------------
# -----------------------
# -----------------------
# EXTRAS BELOW

# plot hr by time of day

# filter so only those with 200+ records but why? why trust above that number?
# is this a place for some sort of effect size and power calculation?
# i.e., if i take 200 consecutive datapoints ranomly from those wtih > 500 
# data points how well does a hr-minimum? estimate the actual hr-minium?
# could also do an adjustment towards the mean when fewer and fewer data
# points, or fewer and fewer non-active datapoints (again would need activity
# data for that).

# wow, crazy oscillations
# other ways to plot?
sns.relplot(x='date', y='resting_hr_5min', data=df_resting_hr_by_day, kind='line')  # the ci takes forever
sns.relplot(x='date', y='resting_hr_week', data=df_resting_hr_by_day, kind='line')  # the ci takes forever
g = sns.relplot(x='date', y='resting_hr_month', data=df_resting_hr_by_day, kind='line')  # the ci takes forever
g.fig.autofmt_xdate()



# -----------------------------------------------------------------------------
# merge continuous hr and daily ratings
# be careful here -- i mapped these ratings to date_sleep, not to date.

date_to_alcohol_dict = dict(zip(df_daily_qs['date'], df_daily_qs['alcohol']))
df_hr_continuous_min['alcohol'] = df_hr_continuous_min['date_sleep'].map(date_to_alcohol_dict)
df_hr_continuous_min['alcohol'].unique()

df_daily_qs[df_daily_qs['alcohol'].notnull()]

date_to_subjective_sleep_dict = dict(zip(df_daily_qs['date'], df_daily_qs['sleep']))
df_hr_continuous_min['sleep_subjective'] = df_hr_continuous_min['date_sleep'].map(date_to_subjective_sleep_dict)

sns.countplot('sleep_subjective', data=df_hr_continuous_min)
# find when i changed the sleep scale. and re-scale those days so fits into 1 to 5.
# up until 2/5/17
df_hr_continuous_min[['date', 'date_sleep', 'sleep_subjective']].head()
sns.countplot('sleep_subjective', data=df_hr_continuous_min[df_hr_continuous_min['date_sleep']<'2017-02-25'])
sns.countplot('sleep_subjective', data=df_hr_continuous_min[df_hr_continuous_min['date_sleep']>'2017-02-25'])

#df_hr_continuous_min['sleep_subjective_before'] = np.nan
#df_hr_continuous_min.loc[df_hr_continuous_min['date_sleep']<'2017-02-25', 'sleep_subjective_before'] = df_hr_continuous_min['sleep_subjective']

df_hr_continuous_min['sleep_subjective_rev'] = df_hr_continuous_min['sleep_subjective']
df_hr_continuous_min.loc[(df_hr_continuous_min['date_sleep']<='2017-02-25') & (df_hr_continuous_min['sleep_subjective']==7.), 'sleep_subjective_rev'] = 5
df_hr_continuous_min.loc[(df_hr_continuous_min['date_sleep']<='2017-02-25') & (df_hr_continuous_min['sleep_subjective']==6.), 'sleep_subjective_rev'] = 5
df_hr_continuous_min.loc[(df_hr_continuous_min['date_sleep']<='2017-02-25') & (df_hr_continuous_min['sleep_subjective']==5.), 'sleep_subjective_rev'] = 4
df_hr_continuous_min.loc[(df_hr_continuous_min['date_sleep']<='2017-02-25') & (df_hr_continuous_min['sleep_subjective']==4.), 'sleep_subjective_rev'] = 3
df_hr_continuous_min.loc[(df_hr_continuous_min['date_sleep']<='2017-02-25') & (df_hr_continuous_min['sleep_subjective']==3.), 'sleep_subjective_rev'] = 2
df_hr_continuous_min.loc[(df_hr_continuous_min['date_sleep']<='2017-02-25') & (df_hr_continuous_min['sleep_subjective']==2.), 'sleep_subjective_rev'] = 1
df_hr_continuous_min.loc[(df_hr_continuous_min['date_sleep']<='2017-02-25') & (df_hr_continuous_min['sleep_subjective']==1.), 'sleep_subjective_rev'] = 1

df_hr_continuous_min['sleep_subjective_rev'].unique()
sns.countplot('sleep_subjective_rev', data=df_hr_continuous_min[df_hr_continuous_min['date_sleep']<'2017-02-25'])
sns.countplot('sleep_subjective_rev', data=df_hr_continuous_min[df_hr_continuous_min['date_sleep']>'2017-02-25'])
sns.countplot('sleep_subjective_rev', data=df_hr_continuous_min)


# energy
date_to_energy_dict = dict(zip(df_daily_qs['date'], df_daily_qs['energy']))
df_hr_continuous_min['energy'] = df_hr_continuous_min['date_sleep'].map(date_to_energy_dict)
sns.countplot('energy', data=df_hr_continuous_min[df_hr_continuous_min['date_sleep']<'2017-02-25'])
sns.countplot('energy', data=df_hr_continuous_min[df_hr_continuous_min['date_sleep']>'2017-02-25'])

# annoyed
date_to_annoyed_dict = dict(zip(df_daily_qs['date'], df_daily_qs['annoyed']))
df_hr_continuous_min['annoyed'] = df_hr_continuous_min['date_sleep'].map(date_to_annoyed_dict)
sns.countplot('annoyed', data=df_hr_continuous_min)

# fun
date_to_fun_dict = dict(zip(df_daily_qs['date'], df_daily_qs['fun']))
df_hr_continuous_min['fun'] = df_hr_continuous_min['date_sleep'].map(date_to_fun_dict)
sns.countplot('fun', data=df_hr_continuous_min)


# make sure that these daily ratings are lined up with sleep from the night before
# yeah -- pretty sure it's correct
df_hr_continuous_min[['date', 'date_sleep']][1400:1410]

# for emotion ratings -- these are from the day and sleep from night before
# so the analyses will naturally be looking at effect of hr-sleep on emotion
# if want to look at effect of emotion on sleep, need to lag date here in
# daily ratings and then map onto df_hr_continuous_min

# ------------------------------------------------------------------
# viz a night to help come up wtih hypoths


# clean data --
# deal with outliers hr measurements.
del df_hr_continuous_min['hour']
df_hr_continuous_min.tail()
df_hr_continuous_min.columns

# certain deviation from the min prior if asleep.
# doesn't really work because might have nans before an errant/outlier value

df_hr_continuous_min_to_save = df_hr_continuous_min[['date_time', 'hr', 'sleep_status', 'date', 'date_sleep',
                                             'alcohol', 'sleep_subjective_rev', 'energy', 'fun', 'annoyed']]
df_hr_continuous_min_to_save.rename(columns={'sleep_subjective_rev':'sleep_subjective'}, inplace=True)
df_hr_continuous_min_to_save.tail()
#df_hr_continuous_min_to_save.to_csv('df_hr_for_analyses.csv')


# a certain deviation from the smoothed line?
def clean_hr(df):
    df['hr_rolling'] = df['hr'].rolling(window=30, center=True, min_periods=20).mean()
    df['hr_rolling'] = df['hr_rolling'].interpolate()
    # label as outlier if hr is  +/- 10 beats per min than the smoothed hr_rolling
    df['outlier'] = 0
    df.loc[(df['hr'] > (df['hr_rolling'] + 10)) |
    (df['hr'] < (df['hr_rolling'] - 10)), # & (df['sleep_status'] == 1)
    'outlier'] = 1
    # if hr is outlier, replace with nan
    df['hr_cleaned'] = df['hr']
    df.loc[df['outlier']==1, 'hr_cleaned'] = np.nan
    df[df['outlier']==1].head()
    return df

df_hr_continuous_min = clean_hr(df_hr_continuous_min)



def plot_hr_and_hr_rolling_for_night(df_hr_continuous_min, date, color, resample_minutes, hr_cleaned_or_not):
    # get data from one night
    df_night = df_hr_continuous_min[(df_hr_continuous_min['date_sleep']==date)]
    df_night_asleep = df_night[df_night['sleep_status']==1]
    df_night_asleep['date_temporary_1'] = pd.to_datetime('2017-1-1')
    df_night_asleep['date_temporary_2'] = pd.to_datetime('2017-1-2')
    df_night_asleep['date_time'] = df_night_asleep.index
    df_night_asleep['date_time_temporary'] = np.nan
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour>20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_1'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour<20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_2'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep = df_night_asleep.set_index('date_time_temporary')
    df_night_asleep = df_night_asleep[df_night_asleep.index < '2017-01-02 11:00:00']
    # this adds back minutes that i was awake -- so now the timeline is real time
    # not minutes asleep. it fills these awake minutes with nans, so will see breaks in hr for mins when awake. good.
    df_night_asleep = df_night_asleep.resample(resample_minutes+'min').mean()
    # plot
    ax = plt.figure()
    plt.plot(df_night_asleep.index, df_night_asleep[hr_cleaned_or_not], alpha=.2, color=color)
    plt.plot(df_night_asleep.index, df_night_asleep['hr_rolling'], alpha=.3, color='black')
    plt.ylabel('Heart Rate', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.ylim(40,80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=13)
    plt.title('Heart Rate Over the Course of a Night', fontsize=18)
    ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.autofmt_xdate()
    sns.despine()

# show these plots for presentation
date = '2017-5-27'
color = 'green'
resample_minutes = '1'

# plot #1: this shows wtih hr raw:
plot_hr_and_hr_rolling_for_night(df_hr_continuous_min, date, color, resample_minutes, 'hr')
# this shows with hr without outliers:
plot_hr_and_hr_rolling_for_night(df_hr_continuous_min, date, color, resample_minutes, 'hr_cleaned')


# get rolling hr using hr_cleaned. think this makes sense. it's a better rolling metric
# also, i'm doing it on all hr_cleaned, when awake and asleep. think this is what
# i want. i wouldn't want to create rolling hr metric on only times when asleep
# because then will miss times when hr rising just before an awake spell and clean it up too much, i think?
df_hr_continuous_min['hr_rolling_cleaned'] = df_hr_continuous_min['hr_cleaned'].rolling(window=30, center=True, min_periods=20).mean()

# wait - not sure if should do this. it smooths it out even more.
#df_hr_continuous_min['hr_rolling_cleaned'] = df_hr_continuous_min['hr_rolling_cleaned'].rolling(window=20, center=True, min_periods=5).mean()



# ok, when i resample in function below -- i'd cut points when not sleeping out
# when i resample it makes these null. so shows actual time in
# plot vs. if don't resample i see time asleep but not actual time
# hmmm. what do i want? i'd say i want to plot real time.
# but what about for metrics? should be fine with  resampled too
# since have nans for times when awake. BUT, i'm going to have to just
# select rows that I'm asleep in to compute metrics. so for metrics,
# don't interpolate. or it doesn't matter because the nulls from being
# awake will get cut out anyway.
def plot_hr_smoothed_vs_jagged_over_course_of_night_2(df_hr_continuous_min, date, color, resample_minutes):
    # get data from one night
    df_night = df_hr_continuous_min[(df_hr_continuous_min['date_sleep']==date)]
    df_night_asleep = df_night[df_night['sleep_status']==1]
    df_night_asleep['date_temporary_1'] = pd.to_datetime('2017-1-1')
    df_night_asleep['date_temporary_2'] = pd.to_datetime('2017-1-2')
    df_night_asleep['date_time'] = df_night_asleep.index
    df_night_asleep['date_time_temporary'] = np.nan
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour>20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_1'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour<20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_2'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep = df_night_asleep.set_index('date_time_temporary')
    df_night_asleep = df_night_asleep[df_night_asleep.index < '2017-01-02 11:00:00']
    # create rolling metric
    #df_night_asleep['hr_rolling_cleaned'] = df_night_asleep['hr_rolling_cleaned'].interpolate()  # this isn't really necessary when have min_periods set above
    # this adds back minutes that i was awake -- so now the timeline is real time
    # not minutes asleep. it fills these awake minutes with nans, so will see breaks in hr for mins when awake. good.
    df_night_asleep = df_night_asleep.resample(resample_minutes+'min').mean()
    # plot
    ax = plt.figure()
    plt.plot(df_night_asleep.index, df_night_asleep['hr_cleaned'], alpha=.2, color=color)
    plt.plot(df_night_asleep.index, df_night_asleep['hr_rolling_cleaned'], alpha=.3, color='black')
    plt.ylabel('Heart Rate', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.ylim(40,80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=13)
    plt.title('Heart Rate Over the Course of a Night', fontsize=18)
    ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.autofmt_xdate()
    sns.despine()


def plot_hr_over_course_of_night(df_hr_continuous_min, date, color, resample_minutes):
    # get data from one night
    df_night = df_hr_continuous_min[(df_hr_continuous_min['date_sleep']==date)]
    df_night_asleep = df_night[df_night['sleep_status']==1]
    #df_night_asleep = df_night[ (df_night['sleep_status']==1) | (df_night['sleep_status']==2) ]
    # strip datetime of date portion, leave time
    df_night_asleep['date_temporary_1'] = pd.to_datetime('2017-1-1')
    df_night_asleep['date_temporary_2'] = pd.to_datetime('2017-1-2')
    df_night_asleep['date_time'] = df_night_asleep.index
    df_night_asleep['date_time_temporary'] = np.nan
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour>20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_1'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour<20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_2'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    #df_night_asleep['date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep = df_night_asleep.set_index('date_time_temporary')
    df_night_asleep = df_night_asleep[df_night_asleep.index < '2017-01-02 11:00:00']
    # create rolling metric -
    #df_night_asleep['hr_rolling_cleaned'] = df_night_asleep['hr_cleaned'].rolling(window=30, center=True, min_periods=20).mean()
    #df_night_asleep['hr_rolling_cleaned'] = df_night_asleep['hr_rolling_cleaned'].interpolate()  # this isn't really necessary when have min_periods set above
    # this adds back minutes that i was awake -- so now the timeline is real time
    # not minutes asleep. it fills these awake minutes with nans, so will see breaks in hr for mins when awake. good.
    df_night_asleep = df_night_asleep.resample(resample_minutes+'min').mean()
    # plot
    ax = plt.figure()
#    plt.plot(df_night_asleep.index, df_night_asleep['hr_cleaned'], alpha=.2, color=color)
    plt.plot(df_night_asleep.index, df_night_asleep['hr_rolling_cleaned'], alpha=.3, color='black')
    #plt.plot(df_night_asleep.index, df_night_asleep['hr_rolling_cleaned'], alpha=.15, color=color)
    plt.ylabel('Heart Rate', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    #plt.xlim('2017-01-01 22:00:00', '2017-01-02 11:00:00')
    plt.ylim(40,80)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=13)
    plt.title('Heart Rate Over the Course of a Night', fontsize=18)
    ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.autofmt_xdate()
    sns.despine()




date = '2017-4-28'
date = '2017-4-29'
date = '2017-4-30'
date = '2017-5-1'
date = '2017-5-2'
date = '2017-5-8'
date = '2016-12-1'
date = '2016-12-3'
date = '2016-12-4'
date = '2016-12-5'  # here the outlier isn't removed. why? # should i be removing hr that are x beats faster than prior hr_rolling (and not just the immediate hr rolling?)
date = '2016-12-6'
date = '2016-12-7'
date = '2016-12-8'
date = '2017-6-4'
date = '2017-4-27'

date = '2017-5-27'
color = 'green'
resample_minutes = '1'

# plot #2: this shows with hr without outliers and smoothed line creatd without these outliers
plot_hr_smoothed_vs_jagged_over_course_of_night_2(df_hr_continuous_min, date, color, resample_minutes)

plot_hr_over_course_of_night(df_hr_continuous_min, date, color, resample_minutes)



dates = pd.date_range('2017-8-1', '2017-8-5', freq='D')
dates = pd.date_range('2017-7-20', '2017-7-31', freq='D')
dates = pd.date_range('2017-06-1', '2017-8-7', freq='D')
dates = pd.date_range('2017-05-16', '2017-06-16', freq='D')
dates = pd.date_range('2017-7-15', '2017-8-7', freq='D')
colors = sns.color_palette('pastel', len(dates))
#colors = ['blue', 'green', 'purple', 'red']


def plot_hr_over_night_for_overlay(df_hr_continuous_min, date, line_color, resample_minutes, alpha_level):
    df_night = df_hr_continuous_min[(df_hr_continuous_min['date_sleep']==date)]
    df_night_asleep = df_night[df_night['sleep_status']==1]
    df_night_asleep['date_temporary_1'] = pd.to_datetime('2017-1-1')
    df_night_asleep['date_temporary_2'] = pd.to_datetime('2017-1-2')
    df_night_asleep['date_time'] = df_night_asleep.index
    df_night_asleep['date_time_temporary'] = np.nan
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour>20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_1'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour<20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_2'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep = df_night_asleep.set_index('date_time_temporary')
    df_night_asleep = df_night_asleep[df_night_asleep.index < '2017-01-02 11:00:00']
    df_night_asleep = df_night_asleep.resample(resample_minutes+'min').mean()
    plt.plot(df_night_asleep.index, df_night_asleep['hr_rolling_cleaned'], alpha=alpha_level, color=line_color)


# overlay
ax = plt.figure()
for date in dates:
    plot_hr_over_night_for_overlay(df_hr_continuous_min, date, 'blue', resample_minutes, .1)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
plt.ylabel('Heart Rate', fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylim(40,80)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.title('Heart Rate Over the Course of a Night', fontsize=18)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
sns.despine()


# overlay and give line diff colors dependingon alcohol
# could be nice to show a month or two overlayed.
ax = plt.figure()
for date in dates:
    #print(date)
    if df_hr_continuous_min[df_hr_continuous_min['date']==date]['alcohol'].values[0] == 1:
        color = 'red'
    elif df_hr_continuous_min[df_hr_continuous_min['date']==date]['alcohol'].values[0] == 0:
        color = 'green'
    elif df_hr_continuous_min[df_hr_continuous_min['date']==date]['alcohol'].values[0] > 1:
        color = 'red'
    else:
        color = 'grey'
    plot_hr_over_night_for_overlay(df_hr_continuous_min, date, color, resample_minutes)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
plt.ylabel('Heart Rate', fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylim(40,80)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.title('Heart Rate Over the Course of a Night', fontsize=18)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
sns.despine()



#ax = plt.figure()
#for date, color in zip(dates, colors):
#    #print(date)
#    if df_hr_continuous_min[df_hr_continuous_min['date']==date]['alcohol'].values[0] == 1:
#        color = 'red'
#    elif df_hr_continuous_min[df_hr_continuous_min['date']==date]['alcohol'].values[0] == 0:
#        color = 'green'
#    elif df_hr_continuous_min[df_hr_continuous_min['date']==date]['alcohol'].values[0] > 1:
#        color = 'purple'
#    else:
#        color = 'grey'
#    plot_hr_over_course_of_night(df_hr_continuous_min, date, color, resample_minutes)
##ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
#ax.autofmt_xdate()
#
#
#ax = plt.figure()
#for date, color in zip(dates, colors):
#    print(date)
#    if df_hr_continuous_min[df_hr_continuous_min['date']==date]['sleep_subjective'].values[0] < 3:
#        color = 'red'
#    elif df_hr_continuous_min[df_hr_continuous_min['date']==date]['sleep_subjective'].values[0] > 3:
#        color = 'green'
#    else:
#        color = 'grey'
#    plot_hr_over_course_of_night(df_hr_continuous_min, date, color, resample_minutes)
#ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
#ax.autofmt_xdate()


df_hr_continuous_min['alcohol']
df_hr_continuous_min['fun']
len(df_hr_continuous_min['date'].unique())

df_size = df_hr_continuous_min.groupby(['date_sleep']).mean().reset_index()
df_size.groupby('alcohol').size()
sns.countplot(df_size['alcohol'], color='red', alpha=.6)
sns.countplot(df_size['fun'], color='red', alpha=.6)
sns.countplot(df_size['sleep_subjective_rev'], color='red', alpha=.6)

df_hr_continuous_min[df_hr_continuous_min['date']=='2017-08-08']['alcohol'].value_counts()
df_hr_continuous_min[df_hr_continuous_min['date']=='2017-05-12']['alcohol'].value_counts()

# ideas to try

# get average lines for diff amounts of alcohol
# for just nights i drink, look at diff types of alcohol

# fft to get frequencies
# deviations from smoothed line (this is really high freq variability)
# min of smoothed line, max of smoothed line
# smoothed line at diff points in night, esp before waking

# ultimately, what do people want to know?
# if they are in a positive mood and have energy

# what do i want to know? want to know how heart data while sleeping looks
# on nights where I think i sleep poorly (subjective sleep rating) and nights
# where i think i slept well. I think my subjective ratings are particuarly
# meaninful for my physical and mental health. if I could see a sign of that,
# some physical counterpart of that in the heart signal while aslee, I think
# that would be really interesting from a personal and scientific standpoint.

# how do i want to look at this?
# could plot two hr time series, one for good and one for bad subj sleep
# is there a way to get variation into this plot, so see viz if variation or what freq of variations differ?
# this seems like some kind of times series technique -- perhaps to overlay different frequencies onto the average line?
# look this up in time series stuff? time series in python search?
# could also calc things like std or variation frome the smoothed line and scatter with subj sleep


# ---------------------------------------------------------
# one average line for alcohol vs. avg line for no alcohol
df_hr_continuous_min['alcohol_dichot'] = np.nan
df_hr_continuous_min.loc[df_hr_continuous_min['alcohol']>0, 'alcohol_dichot'] = 1
df_hr_continuous_min.loc[df_hr_continuous_min['alcohol']==0, 'alcohol_dichot'] = 0
df_hr_continuous_min['alcohol_dichot'].value_counts()
#df_hr_continuous_min.groupby('date_time')['hr'].mean()

df_hr_continuous_min['alcohol'].unique()
sns.countplot(df_hr_continuous_min['alcohol'])
df_hr_continuous_min['alcohol_binned'] = df_hr_continuous_min['alcohol']
df_hr_continuous_min['alcohol_binned'].replace(3, 2, inplace=True)
df_hr_continuous_min['alcohol_binned'].replace(4, 2, inplace=True)
df_hr_continuous_min['alcohol_binned'].replace(5, 2, inplace=True)
sns.countplot(df_hr_continuous_min['alcohol_binned'])

df_hr_continuous_min['sleep_subjective_rev'].unique()
sns.countplot(df_hr_continuous_min['sleep_subjective_rev'])

df_hr_continuous_min['sleep_binned'] = df_hr_continuous_min['sleep_subjective_rev']
df_hr_continuous_min['sleep_binned'].replace(2, 1, inplace=True)
df_hr_continuous_min['sleep_binned'].replace(4, 5, inplace=True)
sns.countplot(df_hr_continuous_min['sleep_binned'])


df_hr_continuous_min.head()
df_hr_continuous_min['energy'].value_counts()
sns.countplot(df_hr_continuous_min['energy'])


# think i should look up time series stuff and how to get avareage of multiple time series's, etc.
# http://seaborn.pydata.org/generated/seaborn.tsplot.html
# https://chrisalbon.com/python/seaborn_pandas_timeseries_plot.html
# http://twiecki.github.io/blog/2017/03/14/random-walk-deep-net/



# vietnam dates ==============================================================
# Nov 3-12
len(df_hr_continuous_min)
df_hr_continuous_min = df_hr_continuous_min[(df_hr_continuous_min['date']<'2016-11-3') | (df_hr_continuous_min['date']>'2016-11-11')]
len(df_hr_continuous_min)
df_hr_continuous_min['date'].unique()
# vietnam dates ==============================================================


df_hr_continuous_min['date'].unique()
df_hr_continuous_min['date'].min()


# select only when sleeping (and use date_sleep to divide by nights sleep)
df_hr_continuous_asleep = df_hr_continuous_min[df_hr_continuous_min['sleep_status']==1]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['hr_rolling'].notnull()]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['alcohol_dichot'].notnull()]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['date_sleep'].notnull()]

df_hr_continuous_asleep.shape  # (135328, 15)
df_hr_continuous_asleep.head()
df_hr_continuous_asleep.tail()

# create variable that says how many minutes i've been aleep for that night
df_hr_continuous_asleep['consecutive_numbers'] = np.arange(1.,len(df_hr_continuous_asleep)+1)
#df_hr_continuous_asleep['consecutive_numbers_lag'] = df_hr_continuous_asleep.groupby('date_sleep')['consecutive_numbers'].transform(lambda x: x.shift(1))
df_hr_continuous_asleep['sum_minutes_asleep'] = df_hr_continuous_asleep.groupby('date_sleep')['consecutive_numbers'].transform(lambda x: x.count())
df_hr_continuous_asleep['first_consecutive_number_of_group'] = df_hr_continuous_asleep.groupby('date_sleep')['consecutive_numbers'].transform(lambda x: x.head(1))
df_hr_continuous_asleep['minutes_asleep'] = df_hr_continuous_asleep['consecutive_numbers'] - df_hr_continuous_asleep['first_consecutive_number_of_group'] + 1
df_hr_continuous_asleep['minutes_asleep'] = df_hr_continuous_asleep.loc[:, 'minutes_asleep'] - 1
df_hr_continuous_asleep[['minutes_asleep', 'date_sleep', 'hr_rolling', 'alcohol_dichot']][990:1050]

# count backwards from waking up
df_hr_continuous_asleep['minutes_from_waking'] = df_hr_continuous_asleep['sum_minutes_asleep'] - df_hr_continuous_asleep['minutes_asleep']
df_hr_continuous_asleep[['date_sleep', 'minutes_from_waking', 'minutes_asleep']][990:1050]


# says just get nights where i slept more then 350 min and only take the records of the first 350 min so I can plot below
# make this 360 so can say 6 hours
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['sum_minutes_asleep']>350]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['minutes_asleep']<350]

#df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['sum_minutes_asleep']>350]
#df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['minutes_from_waking']<350]

df_hr_continuous_asleep.groupby('date_sleep')['minutes_asleep'].count().min()

df_hr_continuous_asleep['sum_minutes_asleep'].hist()
df_hr_continuous_asleep['minutes_asleep'].hist()


#df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['hr_rolling'].notnull()]
#df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['first_consecutive_number_of_group'].notnull()]
#df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['date_sleep'].notnull()]
#df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['minutes_asleep']<400]

df_hr_continuous_asleep_ordered = df_hr_continuous_asleep.sort_values(by='alcohol', ascending=False)
df_hr_continuous_asleep_ordered['alcohol_dichot_words'] = np.nan
df_hr_continuous_asleep_ordered.loc[df_hr_continuous_asleep_ordered['alcohol_dichot'] == 0, 'alcohol_dichot_words'] = 'No Drinks'
df_hr_continuous_asleep_ordered.loc[df_hr_continuous_asleep_ordered['alcohol_dichot'] == 1, 'alcohol_dichot_words'] = '1+ Drinks'

# show this graph
sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', condition='alcohol_binned', ci=95, n_boot=400, data=df_hr_continuous_asleep_ordered, color=['r', 'orange', 'g'], alpha=.4)  # err_style='boot_traces',
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(['2+ drinks','1 drink','No drinks'], fontsize=15)
sns.despine()

df_hr_continuous_asleep_ordered['alcohol'].replace(5,4, inplace=True)
df_hr_continuous_asleep_ordered['alcohol'].replace(4,3, inplace=True)
#colors = {0: 'purple', 1:'blue', 2:'green', 3:'orange', 4:'red'}
colors = {0: 'purple', 1:'green', 2:'orange', 3:'red'}
#colors = sns.color_palette("hls", 8)
#colors = sns.color_palette('Blues', 7)
sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', color=colors, alpha=.3, linewidth=3, condition='alcohol', ci=5, data=df_hr_continuous_asleep_ordered)
plt.legend([3,2,1,0], title='Alcohol', fontsize=15)
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()

sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', color={'No Drinks':'green', '1+ Drinks':'red'}, alpha=.5, condition='alcohol_dichot_words', ci=95, data=df_hr_continuous_asleep_ordered)
plt.legend(title='', fontsize=15)
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()

sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', condition='alcohol_binned', ci=95, data=df_hr_continuous_asleep)
plt.legend(title='Alcohol', fontsize=12)
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()


# re-order these for subj sleep?
df_hr_continuous_asleep_ordered_subjective = df_hr_continuous_asleep.sort_values(by='sleep_subjective_rev', ascending=False)

sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', condition='sleep_subjective_rev', ci=95, data=df_hr_continuous_asleep_ordered_subjective)
plt.legend(title='Subjective Sleep', fontsize=12)
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()

sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', condition='sleep_binned', ci=95, data=df_hr_continuous_asleep_ordered_subjective)
plt.legend(title='Subjective Sleep', fontsize=12)
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()

# i used 90% CI here to show diff towards end of night better -- probably even better with more minutes included?
sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', condition='sleep_binned', ci=90, data=df_hr_continuous_asleep_ordered_subjective[df_hr_continuous_asleep_ordered_subjective['sleep_binned']!=3])
plt.legend(title='Subjective Sleep', fontsize=12)
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()

sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', condition='sleep_binned', color=['green', 'red'], alpha=.7, ci=90, data=df_hr_continuous_asleep_ordered_subjective[(df_hr_continuous_asleep_ordered_subjective['sleep_binned']!=3) & (df_hr_continuous_asleep_ordered_subjective['alcohol']<2)])
plt.legend(title='Subjective Sleep', fontsize=12)
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(['good', 'bad'], fontsize=18, title='Subjective Sleep')
sns.despine()


sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', condition='sleep_binned', color=['green', 'red'], alpha=.7, ci=90, data=df_hr_continuous_asleep_ordered_subjective[(df_hr_continuous_asleep_ordered_subjective['sleep_binned']!=3) & (df_hr_continuous_asleep_ordered_subjective['alcohol']>=2)])
plt.legend(title='Subjective Sleep', fontsize=12)
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(['good', 'bad'], fontsize=18, title='Subjective Sleep')
sns.despine()

sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', condition='sleep_binned', err_style='boot_traces', ci=95, n_boot=400, data=df_hr_continuous_asleep_ordered_subjective[df_hr_continuous_asleep_ordered_subjective['sleep_binned']!=3])
plt.legend(title='Subjective Sleep', fontsize=12)
plt.xlabel('Number of Minutes Asleep', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
sns.despine()

# should re-do the sleep graphs but count backwards - i.e., 1 min from waking up i the mornig, 2 min from waking up
# do i see subjective sleep differentiating from waking point but less and less as go back in time?

df_hr_continuous_asleep = df_hr_continuous_min[df_hr_continuous_min['sleep_status']==1]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['hr_rolling'].notnull()]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['alcohol_dichot'].notnull()]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['date_sleep'].notnull()]
# create variable that says how many minutes i've been aleep for that night
df_hr_continuous_asleep['consecutive_numbers'] = np.arange(1.,len(df_hr_continuous_asleep)+1)
#df_hr_continuous_asleep['consecutive_numbers_lag'] = df_hr_continuous_asleep.groupby('date_sleep')['consecutive_numbers'].transform(lambda x: x.shift(1))
df_hr_continuous_asleep['sum_minutes_asleep'] = df_hr_continuous_asleep.groupby('date_sleep')['consecutive_numbers'].transform(lambda x: x.count())
df_hr_continuous_asleep['first_consecutive_number_of_group'] = df_hr_continuous_asleep.groupby('date_sleep')['consecutive_numbers'].transform(lambda x: x.head(1))
df_hr_continuous_asleep['minutes_asleep'] = df_hr_continuous_asleep['consecutive_numbers'] - df_hr_continuous_asleep['first_consecutive_number_of_group'] + 1
df_hr_continuous_asleep['minutes_asleep'] = df_hr_continuous_asleep.loc[:, 'minutes_asleep'] - 1
# count backwards from waking up
df_hr_continuous_asleep['minutes_from_waking'] = df_hr_continuous_asleep['sum_minutes_asleep'] - df_hr_continuous_asleep['minutes_asleep']

# says just get nights where i slept more then 350 min and only take the records of the first 350 min so I can plot below
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['sum_minutes_asleep']>350]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['minutes_from_waking']<350]


df_hr_continuous_asleep_ordered_from_waking = df_hr_continuous_asleep.sort_values(by='alcohol', ascending=False)
df_hr_continuous_asleep_ordered_from_waking['alcohol_dichot_words'] = np.nan
df_hr_continuous_asleep_ordered_from_waking.loc[df_hr_continuous_asleep_ordered_from_waking['alcohol_dichot'] == 0, 'alcohol_dichot_words'] = 'No Drinks'
df_hr_continuous_asleep_ordered_from_waking.loc[df_hr_continuous_asleep_ordered_from_waking['alcohol_dichot'] == 1, 'alcohol_dichot_words'] = '1+ Drinks'


# show this graph
sns.tsplot(time='minutes_from_waking', value='hr_rolling', unit='date_sleep', condition='alcohol_binned', ci=95, n_boot=400, data=df_hr_continuous_asleep_ordered_from_waking, color=['r', 'orange', 'g'], alpha=.4)  # err_style='boot_traces',
plt.xlabel('Number of Minutes from Waking', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(45,75)  # 65
plt.legend(['2+ drinks','1 drink','No drinks'], fontsize=15) #
plt.gca().invert_xaxis()
sns.despine()


# re-order these for subj sleep
df_hr_continuous_asleep_ordered_subjective = df_hr_continuous_asleep.sort_values(by='sleep_subjective_rev', ascending=False)

sns.tsplot(time='minutes_from_waking', value='hr_rolling', unit='date_sleep', condition='sleep_binned', err_style='boot_traces', ci=95, n_boot=400, data=df_hr_continuous_asleep_ordered_subjective[df_hr_continuous_asleep_ordered_subjective['sleep_binned']!=3])
plt.legend(title='Subjective Sleep', fontsize=12)
plt.xlabel('Number of Minutes from Waking', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.gca().invert_xaxis()
sns.despine()

sns.tsplot(time='minutes_from_waking', value='hr_rolling', unit='date_sleep', condition='sleep_binned', ci=95, data=df_hr_continuous_asleep_ordered_subjective[df_hr_continuous_asleep_ordered_subjective['sleep_binned']!=3])
plt.legend(title='Subjective Sleep', fontsize=12)
plt.xlabel('Number of Minutes from Waking', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.gca().invert_xaxis()
sns.despine()

# present for subj sleep w 0-1 drinks
sns.tsplot(time='minutes_from_waking', value='hr_rolling', unit='date_sleep', condition='sleep_binned', ci=95, data=df_hr_continuous_asleep_ordered_subjective[(df_hr_continuous_asleep_ordered_subjective['sleep_binned']!=3) & (df_hr_continuous_asleep_ordered_subjective['alcohol_binned']<2)], color=['g', 'r'], alpha=.5)
plt.xlabel('Number of Minutes from Waking', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(45,61)
plt.gca().invert_xaxis()
plt.legend(['Good Sleep', 'Bad Sleep'], fontsize=15)
sns.despine()

# present with 2+ drinks, subjective sleep doesn't match up
sns.tsplot(time='minutes_from_waking', value='hr_rolling', unit='date_sleep', condition='sleep_binned', ci=95, data=df_hr_continuous_asleep_ordered_subjective[(df_hr_continuous_asleep_ordered_subjective['sleep_binned']!=3) & (df_hr_continuous_asleep_ordered_subjective['alcohol_binned']>=2)], color=['g', 'r'], alpha=.5)
plt.xlabel('Number of Minutes from Waking', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(45,61)
plt.gca().invert_xaxis()
plt.legend(['Good Sleep', 'Bad Sleep'], fontsize=15)
sns.despine()

sns.tsplot(time='minutes_from_waking', value='hr_rolling', unit='date_sleep', condition='sleep_subjective_rev', ci=5, data=df_hr_continuous_asleep_ordered_subjective[(df_hr_continuous_asleep_ordered_subjective['alcohol_binned']<2)],  alpha=.5)  # color=['g', 'r'],
plt.legend(title='Subjective Sleep', fontsize=12)
plt.xlabel('Number of Minutes from Waking', fontsize=18)
plt.ylabel('Heart Rate', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.gca().invert_xaxis()
#plt.legend(['Good Sleep', 'Bad Sleep'], fontsize=15)
sns.despine()





df_hr_continuous_asleep.to_csv('hr_continuous_asleep_work.csv')

#sns.tsplot(time='minutes_from_waking', value='hr_rolling', unit='date_sleep', condition='sleep_subjective_rev', ci=95, data=df_hr_continuous_asleep[(df_hr_continuous_asleep['sleep_subjective_rev']==1) | (df_hr_continuous_asleep['sleep_subjective_rev']==5)])
#plt.legend(title='Subjective Sleep', fontsize=12)
#plt.xlabel('Number of Minutes from Waking', fontsize=18)
#plt.ylabel('Heart Rate', fontsize=18)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#plt.gca().invert_xaxis()
#sns.despine()



dftest = pd.DataFrame({'minutes_asleep':[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,6,7,8,9],
                       'date_sleep':[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,4,4,4,4],
                        'hr_rolling':[9,7,8,5,6,4,5,6,7,4,3,4,5,6,8,8,8,9,0,4,4,5,6,5],
                        'alcohol_dichot':[1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2]})

sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', condition='alcohol_dichot', data=dftest)
sns.tsplot(time='minutes_asleep', value='hr_rolling', unit='date_sleep', data=dftest)



# ------------------------------------------------------------------
# Interesting to do a cluster anys -- cluster
# nights of hr based on a bunch of variables. this might be a nice thing
# to show to other people/friends? and then show the daily ratings for
# each cluster? vs. other way around -- show hr metrics for each level
# of a daily rating. think about -- reasons that these two approaches
# are better or worse for certain things? the cluster anys might help
# me see if how many times of sleep i have. that's kind of cool. could
# say there are really only two different patterns. or 3. interesting.


df_hr_continuous_asleep = df_hr_continuous_min[df_hr_continuous_min['sleep_status']==1]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['hr_rolling'].notnull()]
#df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['alcohol_dichot'].notnull()]
df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['date_sleep'].notnull()]
df_hr_continuous_asleep['consecutive_numbers'] = np.arange(1.,len(df_hr_continuous_asleep)+1)
df_hr_continuous_asleep['sum_minutes_asleep'] = df_hr_continuous_asleep.groupby('date_sleep')['consecutive_numbers'].transform(lambda x: x.count())
df_hr_continuous_asleep['first_consecutive_number_of_group'] = df_hr_continuous_asleep.groupby('date_sleep')['consecutive_numbers'].transform(lambda x: x.head(1))
df_hr_continuous_asleep['minutes_asleep'] = df_hr_continuous_asleep['consecutive_numbers'] - df_hr_continuous_asleep['first_consecutive_number_of_group'] + 1
df_hr_continuous_asleep['minutes_asleep'] = df_hr_continuous_asleep.loc[:, 'minutes_asleep'] - 1
df_hr_continuous_asleep['minutes_from_waking'] = df_hr_continuous_asleep['sum_minutes_asleep'] - df_hr_continuous_asleep['minutes_asleep']

df_hr_continuous_asleep.groupby('date_sleep').size().hist(alpha=.6)

df_hr_continuous_asleep = df_hr_continuous_asleep[df_hr_continuous_asleep['sum_minutes_asleep']>200]

df_hr_continuous_asleep.groupby('date_sleep').size().hist(bins=12, alpha=.5)
plt.grid(False)


# get deviations from smoothed lines - substract each hr from line, then sq all (squared deviations) and get mean
df_hr_continuous_asleep['hr_deviation_from_smoothed_hr'] = np.power((df_hr_continuous_asleep['hr_rolling_cleaned'] - df_hr_continuous_asleep['hr_cleaned']), 2)

df_mean_metrics = df_hr_continuous_asleep.groupby('date_sleep')[['hr_rolling_cleaned', 'hr_cleaned', 'outlier', 'sum_minutes_asleep', 'hr_deviation_from_smoothed_hr', 'alcohol_binned']].mean()
df_mean_metrics.columns = ['hr_rolling_mean', 'hr_mean', 'outlier_mean', 'minutes_asleep_mean', 'hr_deviation_from_smoothed_mean', 'alcohol_binned']

df_std_metrics = df_hr_continuous_asleep.groupby('date_sleep')[['hr_rolling_cleaned', 'hr_cleaned', 'hr_deviation_from_smoothed_hr']].std()
df_std_metrics.columns = ['hr_rolling_std', 'hr_std', 'hr_deviation_from_smoothed_std']

df_min_metrics = df_hr_continuous_asleep.groupby('date_sleep')[['hr_rolling_cleaned']].min()
df_min_metrics.columns = ['hr_rolling_min']

df_max_metrics = df_hr_continuous_asleep.groupby('date_sleep')[['hr_rolling_cleaned']].max()
df_max_metrics.columns = ['hr_rolling_max']

df_corr = df_hr_continuous_asleep.groupby('date_sleep')[['minutes_asleep', 'hr_rolling_cleaned']].corr().reset_index()
df_corr = df_corr[df_corr['level_1']=='minutes_asleep']
df_corr = df_corr.set_index('date_sleep')
df_corr = df_corr[['hr_rolling_cleaned']]
df_corr.columns = ['hr_slope']

# take out days when in vietnam in other time zone

# fourier/frequencies

# times awake
df_time_start_end_sleep = df_hr_continuous_asleep.groupby('date_sleep')[['date_time']].apply(lambda x: x.max() - x.min())
df_time_start_end_sleep['minutes_asleep'] = df_time_start_end_sleep['date_time'] / np.timedelta64(1, 'm')  # then subtract this from sum_minutes_asleep to get min awake
#df_time_start_end_sleep.columns = ['minutes_asleep']

print(len(df_min_metrics), len(df_mean_metrics), len(df_corr), len(df_time_start_end_sleep))


df_nightly_metrics_joined = df_mean_metrics.join(df_std_metrics)
df_nightly_metrics_joined = df_nightly_metrics_joined.join(df_min_metrics)
df_nightly_metrics_joined = df_nightly_metrics_joined.join(df_max_metrics)
df_nightly_metrics_joined = df_nightly_metrics_joined.join(df_corr)
df_nightly_metrics_joined = df_nightly_metrics_joined.join(df_time_start_end_sleep[['minutes_asleep']])
df_subj_sleep = df_hr_continuous_asleep.groupby('date_sleep')[['sleep_subjective_rev']].mean()
df_nightly_metrics_joined = df_nightly_metrics_joined.join(df_subj_sleep)

df_nightly_metrics_joined['minutes_awake'] = df_nightly_metrics_joined['minutes_asleep'] - df_nightly_metrics_joined['minutes_asleep_mean']
df_nightly_metrics_joined['minutes_awake'] = df_nightly_metrics_joined['minutes_awake']/df_nightly_metrics_joined['minutes_asleep']

df_nightly_metrics_joined.head()
df_nightly_metrics_joined.dtypes
df_nightly_metrics_joined.columns

variables_to_cluster_list = ['hr_rolling_mean', 'hr_mean', 'outlier_mean',
'minutes_asleep_mean', 'hr_deviation_from_smoothed_mean', 'hr_rolling_std',
'hr_std', 'hr_deviation_from_smoothed_std', 'hr_rolling_min', 'hr_rolling_max',
'hr_slope', 'minutes_asleep', 'minutes_awake', 'sleep_subjective_rev']

# yes, need to standardize! any reason to do 0-1 standardizing instead?
for variable in variables_to_cluster_list:
    df_nightly_metrics_joined[variable+'_z'] = (df_nightly_metrics_joined[variable] - df_nightly_metrics_joined[variable].mean()) / df_nightly_metrics_joined[variable].std()
    print(np.round(df_nightly_metrics_joined[variable+'_z'].mean(), 3))

# 0-1 max-min scaling
for variable in variables_to_cluster_list:
    df_nightly_metrics_joined[variable+'_z'] = (df_nightly_metrics_joined[variable] - df_nightly_metrics_joined[variable].min()) / (df_nightly_metrics_joined[variable].max() - df_nightly_metrics_joined[variable].min() )
    print(np.round(df_nightly_metrics_joined[variable+'_z'].mean(), 3))

#for variable in variables_to_cluster_list:
#    print(len(df_nightly_metrics_joined[df_nightly_metrics_joined[variable].isnull()]))
#    df_nightly_metrics_joined = df_nightly_metrics_joined[df_nightly_metrics_joined[variable].notnull()]


variables_z_to_cluster_list = ['hr_mean_z', 'hr_std_z', 'outlier_mean_z',
'hr_deviation_from_smoothed_mean_z', 'hr_deviation_from_smoothed_std_z',
'hr_rolling_min_z', 'hr_rolling_max_z', 'hr_slope_z', 'minutes_asleep_z',
'minutes_awake_z']


df_nightly_metrics_joined[variables_z_to_cluster_list].shape
df_nightly_metrics_joined[variables_z_to_cluster_list].head()

for col in df_nightly_metrics_joined[variables_z_to_cluster_list].columns:
    print(col)
    print(len(df_nightly_metrics_joined[df_nightly_metrics_joined[col].isnull()]))

for col in df_nightly_metrics_joined[variables_z_to_cluster_list].columns:
    df_nightly_metrics_joined[col].fillna(df_nightly_metrics_joined[col].mean(), inplace=True)


variables_z_to_cluster_list = ['hr_mean_z', 'hr_deviation_from_smoothed_mean_z',
'hr_rolling_min_z', 'hr_rolling_max_z']


# get clustering code from nta project
sns.clustermap(df_nightly_metrics_joined[df_nightly_metrics_joined.index>='2017-6-30'][variables_z_to_cluster_list], yticklabels=False)  # ,  cmap='Accent_r'

sns.clustermap(df_nightly_metrics_joined[variables_z_to_cluster_list], yticklabels=False)  # ,  cmap='Accent_r'

sns.clustermap(df_nightly_metrics_joined[variables_z_to_cluster_list], yticklabels=False)  # ,  cmap='Accent_r'

sns.clustermap(df_nightly_metrics_joined[variables_z_to_cluster_list], yticklabels=False, metric='cosine')  # , cmap='Accent_r', metric='correlation'  cosine


alcohol_to_color_dict = dict(zip(df_nightly_metrics_joined['alcohol_binned'].unique(), ['grey', 'orange', 'green', 'red']))
df_nightly_metrics_joined['alcohol_color'] = df_nightly_metrics_joined['alcohol_binned'].map(alcohol_to_color_dict)
df_nightly_metrics_joined[['alcohol_binned', 'alcohol_color']]
df_nightly_metrics_joined['alcohol_color'].fillna('lightgrey', inplace=True)
sns.clustermap(df_nightly_metrics_joined[variables_z_to_cluster_list], yticklabels=False, row_colors=df_nightly_metrics_joined['alcohol_color'])  # , metric='cosine',

sleep_subj_to_color_dict = dict(zip(df_nightly_metrics_joined['sleep_subjective_rev_z'].unique(), ['lightgrey', 'red', 'green', 'green', 'red', 'lightgrey']))
df_nightly_metrics_joined['sleep_color'] = df_nightly_metrics_joined['sleep_subjective_rev_z'].map(sleep_subj_to_color_dict)
df_nightly_metrics_joined[['sleep_subjective_rev_z', 'sleep_color']]
df_nightly_metrics_joined['sleep_color'].fillna('lightgrey', inplace=True)
sns.clustermap(df_nightly_metrics_joined[variables_z_to_cluster_list], yticklabels=False, metric='cosine', row_colors=df_nightly_metrics_joined['sleep_color'])  # , metric='cosine',



#sns.clustermap(df_nightly_metrics_joined[variables_z_to_cluster_list], z_score=1)  # z_score=1 computes z scores
#sns.clustermap(df_nightly_metrics_joined[variables_z_to_cluster_list], standard_scale=1)  # standard_scale=1 computes 0-1 max in stadardizing

variables_z_to_cluster_list = ['hr_mean_z', 'hr_std_z', 'outlier_mean_z',
'hr_deviation_from_smoothed_mean_z', 'hr_deviation_from_smoothed_std_z',
'hr_rolling_min_z', 'hr_rolling_max_z', 'hr_slope_z', 'minutes_asleep_z',
'minutes_awake_z']

# nice -- use this to see subj sleep ratings corr with vars
# BUT make sure these subjective ratings are matching to the correct day!!!
sns.heatmap(df_nightly_metrics_joined[['sleep_subjective_rev_z']+variables_z_to_cluster_list].corr(), annot=True)
# hr deviation from smoothed line corr most strongly with subj sleep
sns.heatmap(df_nightly_metrics_joined[['alcohol_binned']+variables_z_to_cluster_list].corr(), annot=True)
# hr mean corr most strongly wtih alcohol

sns.clustermap(df_nightly_metrics_joined[variables_z_to_cluster_list].corr(), annot=True)


# think i should cluster using k nearest and plot centroid of each
# to show diff types of hr at night!
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

# inertia curve - to see what k to use?
variables_for_inertia_list = ['hr_mean_z', 'hr_std_z', 'outlier_mean_z',
'hr_deviation_from_smoothed_mean_z', 'hr_rolling_min_z', 'hr_rolling_max_z',
'hr_slope_z']

ks = []
inertias = []
for i in range(1, 31):
    model = KMeans(n_clusters=i).fit(df_nightly_metrics_joined[variables_for_inertia_list])
    ks.append(i)
    inertias.append(model.inertia_)

plt.plot(ks, inertias)
plt.xlabel('Number of Clusters', fontsize=18)
plt.ylabel('Inertia\n(higher = looser clusters)', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks([])
plt.axvline(2, linestyle='--', color='orange', alpha=.7, linewidth=1)
plt.axvline(4, linestyle='--', color='orange', alpha=.7, linewidth=1)
sns.despine()

# could make the argument that two types or that four types of sleep using hr data
# and say in plot what % of nights were this cluster

number_of_clusters = 4
model = KMeans(n_clusters=number_of_clusters, random_state=10).fit(df_nightly_metrics_joined[variables_for_inertia_list])
labels = model.labels_
labels = [label+1 for label in labels]
df_nightly_metrics_joined['cluster'] = labels
df_groupby_clusters = df_nightly_metrics_joined.groupby('cluster').size().reset_index().rename(columns={0:'count'})

#   cluster  count
#        1     71
#        2    152
#        3     62
#        4     14

model.cluster_centers_
len(model.cluster_centers_)  # 4 - the coordinates -- points for each var -- of each cluster
night_distance_from_centroid = pairwise_distances(df_nightly_metrics_joined[variables_for_inertia_list], Y=model.cluster_centers_, metric='euclidean', n_jobs=1)
len(night_distance_from_centroid)  # 299 - how far each nigth is from each of teh four centroids
distances_from_cluster_1 = night_distance_from_centroid[:, 0]  # how far each night is from cluster 1
distances_from_cluster_2 = night_distance_from_centroid[:, 1]  # how far each night is from cluster 2
distances_from_cluster_3 = night_distance_from_centroid[:, 2]  # how far each night is from cluster 3
distances_from_cluster_4 = night_distance_from_centroid[:, 3]  # how far each night is from cluster 4

#minimum_from_cluster_1 = np.min(distances_from_cluster_1)
#minimum_from_cluster_2 = np.min(distances_from_cluster_2)
#minimum_from_cluster_3 = np.min(distances_from_cluster_3)
#minimum_from_cluster_4 = np.min(distances_from_cluster_4)
#min_index_from_cluster_1 = np.argmin(distances_from_cluster_1)
#min_index_from_cluster_2 = np.argmin(distances_from_cluster_2)
#min_index_from_cluster_3 = np.argmin(distances_from_cluster_3)
#min_index_from_cluster_4 = np.argmin(distances_from_cluster_4)

enum_night_distance_from_cluster_1_centroid = list(enumerate(distances_from_cluster_1))
enum_night_distance_from_cluster_2_centroid = list(enumerate(distances_from_cluster_2))
enum_night_distance_from_cluster_3_centroid = list(enumerate(distances_from_cluster_3))
enum_night_distance_from_cluster_4_centroid = list(enumerate(distances_from_cluster_4))

night_distance_from_centroid_1_sorted = sorted(enum_night_distance_from_cluster_1_centroid, key=lambda tup: tup[1])
night_distance_from_centroid_2_sorted = sorted(enum_night_distance_from_cluster_2_centroid, key=lambda tup: tup[1])
night_distance_from_centroid_3_sorted = sorted(enum_night_distance_from_cluster_3_centroid, key=lambda tup: tup[1])
night_distance_from_centroid_4_sorted = sorted(enum_night_distance_from_cluster_4_centroid, key=lambda tup: tup[1])

# get distance to each cluster into df
df_nightly_metrics_joined['distance_from_centroid_cluster_1'] = distances_from_cluster_1
df_nightly_metrics_joined['distance_from_centroid_cluster_2'] = distances_from_cluster_2
df_nightly_metrics_joined['distance_from_centroid_cluster_3'] = distances_from_cluster_3
df_nightly_metrics_joined['distance_from_centroid_cluster_4'] = distances_from_cluster_4

df_nightly_metrics_joined['distance_rank_cluster_1'] = df_nightly_metrics_joined['distance_from_centroid_cluster_1'].rank()
df_nightly_metrics_joined['distance_rank_cluster_2'] = df_nightly_metrics_joined['distance_from_centroid_cluster_2'].rank()
df_nightly_metrics_joined['distance_rank_cluster_3'] = df_nightly_metrics_joined['distance_from_centroid_cluster_3'].rank()
df_nightly_metrics_joined['distance_rank_cluster_4'] = df_nightly_metrics_joined['distance_from_centroid_cluster_4'].rank()

# gives the night closest to each cluster centroid
date_center_cluster_1 = df_nightly_metrics_joined[df_nightly_metrics_joined['distance_rank_cluster_1']==2].index[0]
date_center_cluster_2 = df_nightly_metrics_joined[df_nightly_metrics_joined['distance_rank_cluster_2']==2].index[0]
date_center_cluster_3 = df_nightly_metrics_joined[df_nightly_metrics_joined['distance_rank_cluster_3']==2].index[0]
date_center_cluster_4 = df_nightly_metrics_joined[df_nightly_metrics_joined['distance_rank_cluster_4']==2].index[0]

df_nightly_metrics_joined.dtypes
df_nightly_metrics_joined.head()

date = date_center_cluster_1
color = 'green'
resample_minutes = '1'

plot_hr_smoothed_vs_jagged_over_course_of_night_2(df_hr_continuous_min, date_center_cluster_1, color, resample_minutes)
plot_hr_smoothed_vs_jagged_over_course_of_night_2(df_hr_continuous_min, date_center_cluster_2, color, resample_minutes)
plot_hr_smoothed_vs_jagged_over_course_of_night_2(df_hr_continuous_min, date_center_cluster_3, color, resample_minutes)
plot_hr_smoothed_vs_jagged_over_course_of_night_2(df_hr_continuous_min, date_center_cluster_4, color, resample_minutes)


# plot multiple from the same cluster
dates_list = []
for rank in [1,2,3,4,5]:
    date_center_cluster = df_nightly_metrics_joined[df_nightly_metrics_joined['distance_rank_cluster_4']==rank].index[0]
    dates_list.append(date_center_cluster)
# overlay
color = 'blue'
ax = plt.figure()
for date in dates_list:
    plot_hr_over_night_for_overlay(df_hr_continuous_min, date, color, resample_minutes, .3)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
plt.ylabel('Heart Rate', fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylim(40,80)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.title('Heart Rate Over the Course of a Night', fontsize=18)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
sns.despine()


# plot from different clusters
dates = [date_center_cluster_1, date_center_cluster_2, date_center_cluster_3, date_center_cluster_4]
colors = ['red', 'blue', 'green', 'orange']
ax = plt.figure()
for date, color in zip(dates, colors):
    plot_hr_over_night_for_overlay(df_hr_continuous_min, date, color, resample_minutes, .4)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
plt.ylabel('Heart Rate', fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylim(40,80)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.title('Heart Rate Over the Course of a Night', fontsize=18)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
sns.despine()

# would be intersting to see these really smoothed. i think would help.
def plot_hr_over_night_extra_smooth_for_overlay(df_hr_continuous_min, date, line_color, resample_minutes, alpha_level, cluster):
    df_night = df_hr_continuous_min[(df_hr_continuous_min['date_sleep']==date)]
    df_night_asleep = df_night[df_night['sleep_status']==1]
    df_night_asleep['date_temporary_1'] = pd.to_datetime('2017-1-1')
    df_night_asleep['date_temporary_2'] = pd.to_datetime('2017-1-2')
    df_night_asleep['date_time'] = df_night_asleep.index
    df_night_asleep['date_time_temporary'] = np.nan
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour>20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_1'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour<20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_2'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep = df_night_asleep.set_index('date_time_temporary')
    df_night_asleep = df_night_asleep[df_night_asleep.index < '2017-01-02 11:00:00']
    df_night_asleep = df_night_asleep.resample(resample_minutes+'min').mean()
    # smooth more
    df_night_asleep['hr_rolling_cleaned'] = df_night_asleep['hr_rolling_cleaned'].interpolate()  # this isn't really necessary when have min_periods set above
    df_night_asleep['hr_rolling_cleaned'] = df_night_asleep['hr_rolling_cleaned'].rolling(window=20, center=True, min_periods=5).mean()
    plt.plot(df_night_asleep.index, df_night_asleep['hr_rolling_cleaned'], alpha=alpha_level, color=line_color, label=cluster)

def plot_hr_over_night_raw_for_overlay(df_hr_continuous_min, date, line_color, resample_minutes, alpha_level, cluster):
    df_night = df_hr_continuous_min[(df_hr_continuous_min['date_sleep']==date)]
    df_night_asleep = df_night[df_night['sleep_status']==1]
    df_night_asleep['date_temporary_1'] = pd.to_datetime('2017-1-1')
    df_night_asleep['date_temporary_2'] = pd.to_datetime('2017-1-2')
    df_night_asleep['date_time'] = df_night_asleep.index
    df_night_asleep['date_time_temporary'] = np.nan
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour>20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_1'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep.loc[df_night_asleep['date_time'].dt.hour<20, 'date_time_temporary'] = pd.to_datetime(df_night_asleep['date_temporary_2'].astype(str) + ' ' + df_night_asleep['date_time'].dt.hour.astype(str) + ':' + df_night_asleep['date_time'].dt.minute.astype(str))
    df_night_asleep = df_night_asleep.set_index('date_time_temporary')
    df_night_asleep = df_night_asleep[df_night_asleep.index < '2017-01-02 11:00:00']
    df_night_asleep = df_night_asleep.resample(resample_minutes+'min').mean()
    df_night_asleep['hr_cleaned'] = df_night_asleep['hr_cleaned'].interpolate()  # this isn't really necessary when have min_periods set above
    plt.plot(df_night_asleep.index, df_night_asleep['hr_cleaned'], alpha=alpha_level, linewidth=1, color=line_color, label=cluster)  # label=cluster

# plot multiple from the same cluster
dates_list = []
for rank in list(range(1,11)):
    date_center_cluster = df_nightly_metrics_joined[df_nightly_metrics_joined['distance_rank_cluster_4']==rank].index[0]
    dates_list.append(date_center_cluster)
# overlay
color = 'black'
ax = plt.figure()
for date in dates_list:
    plot_hr_over_night_extra_smooth_for_overlay(df_hr_continuous_min, date, color, resample_minutes, .15)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
plt.ylabel('Heart Rate', fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylim(40,80)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
plt.title('Heart Rate Over the Course of a Night', fontsize=18)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
sns.despine()


# plot from different clusters
# plot with just raw, alpha=.5, then with smoothed over and alpha of raw = .2
dates_list = []
rank = 2
cluster_list = [1,2,3,4]
for cluster in cluster_list:
    date_center_cluster = df_nightly_metrics_joined[df_nightly_metrics_joined['distance_rank_cluster_'+str(cluster)]==rank].index[0]
    dates_list.append(date_center_cluster)
colors = ['red', 'blue', 'green', 'orange']
ax = plt.figure()
for date, color, cluster in zip(dates_list, colors, cluster_list):
    plot_hr_over_night_extra_smooth_for_overlay(df_hr_continuous_min, date, color, resample_minutes, .5, cluster)
    plot_hr_over_night_raw_for_overlay(df_hr_continuous_min, date, color, resample_minutes, .2, '')  # using '' for cluster worked! elim putting it into the legend
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
plt.ylabel('Heart Rate', fontsize=18)
plt.xlabel('Time', fontsize=18)
plt.ylim(40,90)
plt.yticks(fontsize=15)
plt.xticks(fontsize=13)
leg = plt.legend(title='Cluster:', fontsize=15)
leg.get_title().set_fontsize(16)
plt.title('Heart Rate Over the Course of a Night', fontsize=18)
ax.axes[0].xaxis.set_major_formatter(DateFormatter('%H:%M'))
ax.autofmt_xdate()
sns.despine()

# plot this but in 4 grids using first through fourth from centroid


# don't really see clear clusters here
sns.clustermap(df_nightly_metrics_joined[variables_for_inertia_list], yticklabels=False)  # ,  cmap='Accent_r' , metric='cosine',











