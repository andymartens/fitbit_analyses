# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 16:23:34 2016

@author: charlesmartens
"""


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

df_hr_awake = df_hr[-df_hr['date_time'].isin(sleep_datetime_list)]
df_hr_sleep = df_hr[df_hr['date_time'].isin(sleep_datetime_list)]

df_hr_awake.shape  # (599287, 4)
df_hr_sleep.shape  # (345425, 4)

#df_hr_awake.to_pickle('df_hr_awake.pkl')
#df_hr_sleep.to_pickle('df_hr_sleep.pkl')

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
plt.xlabel('Hour', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
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

df_sleep['hr'] = df_sleep['date_time'].map(date_time_to_hr_dict)
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

sleep_hours_list = [18, 19, 20,21,22,23,24,0,1,2,3,4,5,6,7,8,9,10,11]
df_sleep_8_to_11 = df_sleep[df_sleep['hour'].isin(sleep_hours_list)]
len(df_sleep_8_to_11) / len(df_sleep)  # 98.7% of the file remains

# give new date that relates to the day the sleep started. 
# e.g., if tue is july 1, then that night is sleep associated with july 1

# give new date that relates to the day the sleep started. 
# e.g., if tue is july 1, then that night is sleep associated with july 1
# to do this, substract 12 hours from the date. so that 11:59am that morning
# becomes 11:49pm of the prior day. and the earlist time, 18:00 (6pm) would
# become 6am. so now that whole time frame is labeled as one date, teh date
# when it started. and then just leave the date

df_sleep_8_to_11['date_sleep'] = df_sleep_8_to_11['date_time'] - timedelta(hours=12)
df_sleep_8_to_11['date_sleep'] = pd.to_datetime(df_sleep_8_to_11['date_sleep'].dt.date)
df_sleep_8_to_11.head()

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
# 41 days have 2+ sleep sessions

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

# plot rolling vs raw on a day
hr_variable = 'hr_interpolate'  # 'hr_interpolate_clean'
hr_variable = 'hr_interpolate_clean'  # 'hr_interpolate'
dates_list = df_sleep_8_to_11_resampled['date_sleep'].unique()
date = dates_list[600]
date = '2017-5-26'
date = '2018-09-04'
date = '2018-07-18'
date = '2017-05-07'
#date = '2016-10-27'  # can see the low outlier
df_day = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['date_sleep']==date]
plt.plot(df_day['date_time'], df_day[hr_variable], 
         alpha=.4, color='green', linewidth=1)
plt.plot(df_day['date_time'], df_day['hr_rolling_30_min'], 
         alpha=.5, color='grey', linewidth=3)  
plt.grid(axis='y', alpha=.4)
plt.ylim(44,65)

hr_threshold = 40  # 50
print(len(df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['hr_interpolate']<hr_threshold]['date_sleep'].unique()))
print(len(df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['hr_interpolate_clean']<hr_threshold]['date_sleep'].unique()))


# remove vietnam dates - nov 3-12
len(df_sleep_8_to_11_resampled['date_sleep'].unique())  # 699
dates_vietnam_list = pd.date_range('2016-11-02', '2016-11-12')
df_sleep_8_to_11_resampled = df_sleep_8_to_11_resampled[-df_sleep_8_to_11_resampled['date_sleep'].isin(dates_vietnam_list)]
len(df_sleep_8_to_11_resampled['date_sleep'].unique())  # 697


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
fig, ax = plt.subplots(nrows=1, ncols=1)
sns.relplot(x='date_time_bogus', y='hr_interpolate_clean', 
            data=df_sleep_8_to_11_resampled,  # .sample(n=1000), 
            kind='line', ci=95, ax=ax)
# assign locator and formatter for the xaxis ticks.
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
#axis.xaxis.set_major_locator(HourLocator(byhour))
ax.set_xlabel('Time', fontsize=15)
ax.set_ylabel('Heart Rate', fontsize=15)
#ax.set_yticklabels(list(range(50,70,2)))
yticks = list(range(50,71,2))
yticklabels = [str(tick) for tick in yticks]
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.set_xlim('2018-10-01 21:00:00', '2018-10-02 10:00:00')
ax.set_ylim(50,68)
fig.autofmt_xdate()
sns.despine()




# overlay smoothed for many nights
alpha_level = .05
width_for_line = 1
yticks = list(range(40,80,5))
yticklabels = [str(tick) for tick in yticks]
fig, ax = plt.subplots(nrows=1, ncols=1)
for date in df_sleep_8_to_11_resampled['date_sleep'].unique()[:]:
    df_date = df_sleep_8_to_11_resampled[df_sleep_8_to_11_resampled['date_sleep']==date]
    plt.plot(df_date['date_time_bogus'], df_date['hr_rolling_30_min'], 
             alpha=alpha_level, color='green', linewidth=width_for_line)
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
fig.autofmt_xdate()
ax.set_xlabel('Time', fontsize=15)
ax.set_ylabel('Heart Rate', fontsize=15)
ax.set_xlim('2018-10-01 21:00:00', '2018-10-02 11:00:00')
ax.set_ylim(43,82)
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
sns.despine()
# use this to grab a few outliers -- what's going on here?


# ------------
# do some aggregate anys:
# subjective sleep ratings
# alcohol
# fun ratings
# energy ratings (or just fun, if this reflects poorly on me?)

# then viz in ts with ci
















# -----------------------
# -----------------------
# -----------------------
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











