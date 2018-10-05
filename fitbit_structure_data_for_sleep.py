# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 16:23:34 2016

@author: charlesmartens
"""


cd \\chgoldfs\253Broadway\PrivateFiles\amartens\hr\hr_sleep_data\fitbit_data2

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
df_sleep['hr'].fillna(method='ffill', limit=1, inplace=True)
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
sleep_hours_list = [20,21,22,23,24,0,1,2,3,4,5,6,7,8,9,10,11]
df_sleep_8_to_11 = df_sleep[df_sleep['hour'].isin(sleep_hours_list)]
len(df_sleep_8_to_11) / len(df_sleep)  # 98% of the file remains

# give new date that relates to the day the sleep started. 
# e.g., if tue is july 1, then that night is sleep associated with july 1

df_sleep_8_to_11.head()




#df_hr_awake_6_to_8.groupby('date').size().hist(alpha=.6, bins=20)
#plt.axvline(df_hr_awake_6_to_8.groupby('date').size().mean(), linestyle='--')
#plt.grid(False)
#df_datapionts_by_day = df_hr_awake_6_to_8.groupby('date').size().reset_index().rename(columns={0:'size'})
#dates_to_keep_list = df_datapionts_by_day[df_datapionts_by_day['size']>200]['date'].values
#df_hr_awake_6_to_8 = df_hr_awake_6_to_8[df_hr_awake_6_to_8['date'].isin(dates_to_keep_list)]
#len(df_hr_awake_6_to_8['date'].unique())  # 708
#len(dates_to_keep_list)
#df_resting_hr_by_day = df_resting_hr_by_day[df_resting_hr_by_day['date'].isin(dates_to_keep_list)]

























# -----------------------
# plot hr by time of day

# better to do with ts with cis. because boxplot has too many high outliers
# and barplot is better for counts/distributions
df_hr.head()
df_hr['hour'] = df_hr['date_time'].dt.hour
df_hr['time'] = df_hr['date_time'].dt.time

df_hr = df_hr.sort_values(by='date_time')

# i want a ragged df here. because want to compute rolling mean of hr
# for some period of time (e.g., 5 or 10 min) rather than for a number of rows.
# this is kind of decision-making and thinking should include in presentation
df_hr.set_index('date_time', inplace=True)
df_hr.head()

df_hr['hr_rolling_5min'] = df_hr['hr'].rolling(window='10min', min_periods=5).mean()
df_hr.head(20)
df_hr.tail(20)
# when fitbid doesn't get hr for a given minute, the hr rolling won't
# be computed for that minute or the next 4 min. because i set it so 
# it needs 5 conseq min recordings to compute the rolling mean

dv = 'hr'
dv = 'hr_rolling_5min'
iv = 'hour'
iv = 'time'
g = sns.relplot(x=iv, y=dv, data=df_hr, kind='line', ci=None)  # the ci takes forever
plt.ylim(40,80)
#g.fig.autofmt_xdate()
hour_list = list(range(0,26,2))
hour_string_list = [str(hour) for hour in hour_list]
plt.xticks(hour_list, hour_string_list, fontsize=10)

# use hr while awake and 
df_hr['date_time'] = df_hr.index
df_hr_awake = df_hr[-df_hr['date_time'].isin(sleep_datetime_list)]
df_hr_awake.shape
dv = 'hr'
dv = 'hr_rolling_5min'
iv = 'hour'
iv = 'time'
g = sns.relplot(x=iv, y=dv, data=df_hr_awake, kind='line', ci=None)  # the ci takes forever
plt.ylim(40,80)
#g.fig.autofmt_xdate()

# select times just between 8am and 8pm
df_hr_awake_6_to_8 = df_hr_awake[(df_hr_awake['hour']>=5) & (df_hr_awake['hour']<=20)]
g = sns.relplot(x=iv, y=dv, data=df_hr_awake_6_to_8, kind='line', ci=None)  # the ci takes forever
plt.ylim(55,85)

# get minimum for each day as resting hr
df_resting_hr_by_day = df_hr_awake_6_to_8.groupby('date')['hr_rolling_5min'].min().reset_index()  #.rename(columns={'hr_rolling_5min':'resting_hr_5min'})
#if two rollign hr the same, it'll take earlier one here, the first one
# get the time at which tool the min rolling hr
df_resting_hr_merged = pd.merge(df_hr_awake_6_to_8, df_resting_hr_by_day, on=['date', 'hr_rolling_5min'], how='inner')
df_resting_hr_merged = df_resting_hr_merged.drop_duplicates(subset='date')
sns.countplot(x='hour', data=df_resting_hr_merged, color='dodgerblue', alpha=.7)

# what's the association between the minimum hr and hour?
sns.relplot(x='hour', y='hr_rolling_5min', data=df_resting_hr_merged, kind='line')
sns.lmplot(x='hour', y='hr_rolling_5min', data=df_resting_hr_merged, 
           lowess=True, scatter_kws={'alpha':.01})
sns.barplot(x='hour', y='hr_rolling_5min', data=df_resting_hr_merged, 
            color='dodgerblue', alpha=.7)
plt.ylim(40,58)
# maybe seeing a bit lower in early am. would fix by adjusting for rleationship
# betweent time/hour and hr circadian rhythm. or by looking at 9am-8pm. or 
# maybe looking for low outliers.

# look for outliers in df_hr_awake_6_to_8
# by computing diff from prior hr recording. if it was the prior minute
# what's the distribution of these? sd?
#df_hr_awake_6_to_8['min_diff_prior'] = df_hr_awake_6_to_8['date_time'] - df_hr_awake_6_to_8['date_time'].shift(1)
#df_hr_awake_6_to_8['min_diff_prior'] = df_hr_awake_6_to_8['min_diff_prior'] / np.timedelta64(1, 'm') 
#
#df_hr_awake_6_to_8['hr_diff_prior'] = np.nan
#df_hr_awake_6_to_8.loc[df_hr_awake_6_to_8['min_diff_prior']==1, 'hr_diff_prior'] = df_hr_awake_6_to_8['hr'] - df_hr_awake_6_to_8['hr'].shift(1)
#
#df_hr_awake_6_to_8['hr_diff_prior'].hist(alpha=.5, bins=50)
#plt.grid(False)
## some big outliers
## what's outside of 2 or 3 sds?
#df_hr_awake_6_to_8['hr_diff_prior'].mean()
#df_hr_awake_6_to_8['hr_diff_prior'].std()
#
#df_hr_awake_6_to_8['hr_diff_prior'].hist(alpha=.5, bins=50)
#plt.axvline(df_hr_awake_6_to_8['hr_diff_prior'].mean() + df_hr_awake_6_to_8['hr_diff_prior'].std()*2, linestyle='--', color='r', linewidth=1, alpha=.5)
#plt.axvline(df_hr_awake_6_to_8['hr_diff_prior'].mean() - df_hr_awake_6_to_8['hr_diff_prior'].std()*2, linestyle='--', color='r', linewidth=1, alpha=.5)
#plt.grid(False)
#
## could use hr drop of -11.5 as cutoff for what consider an outlier
## in this case, what do i replace it with? moving average didn't work that well here
## replace with avg of prior and next hr
#df_hr_awake_6_to_8.loc[df_hr_awake_6_to_8['hr_diff_prior']<-11.5, 'hr'] = np.nan
#df_hr_awake_6_to_8.loc[df_hr_awake_6_to_8['hr_diff_prior']<-11.5, 'hr'] = (df_hr_awake_6_to_8['hr'].shift(1) + df_hr_awake_6_to_8['hr'].shift(-1)) / 2 

# graph below suggests that moving avg is better. because this way above lookng
# at diff between two conseq hr doesn't get rid of outliers when there are a bunch
# in a row that don't seem to make sense. see graph below of smoothed with actual.
 
# smooth and look for diff between smoothed and raw hr
# then could takt outliers and replace with the moving/rollling avg

# rolling at 30 min to find outliers worked well for me in jup nb
df_hr_awake_6_to_8['hr_rolling_30_min'] = df_hr_awake_6_to_8['hr'].rolling(window=30, min_periods=10, center=True).mean()

day = '2016-10-27'  # can see the low outlier
df_awake_day = df_hr_awake_6_to_8[df_hr_awake_6_to_8['date']==day]
df_awake_day = df_awake_day[100:400]
plt.plot(df_awake_day.index, df_awake_day['hr'], 
         alpha=.5, color='green', linewidth=1.25)
plt.plot(df_awake_day.index, df_awake_day['hr_rolling_30_min'], 
         alpha=.3, color='grey', linewidth=4)  
plt.grid(axis='y', alpha=.4)

# get diff between smoothed and actual hr
df_hr_awake_6_to_8['hr_vs_smoothed_diff'] = df_hr_awake_6_to_8['hr'] - df_hr_awake_6_to_8['hr_rolling_30_min']

df_hr_awake_6_to_8['hr_vs_smoothed_diff'].hist(alpha=.5, bins=50)
plt.axvline(df_hr_awake_6_to_8['hr_vs_smoothed_diff'].mean() + df_hr_awake_6_to_8['hr_vs_smoothed_diff'].std()*2, linestyle='--', color='r', linewidth=1, alpha=.5)
plt.axvline(df_hr_awake_6_to_8['hr_vs_smoothed_diff'].mean() - df_hr_awake_6_to_8['hr_vs_smoothed_diff'].std()*2, linestyle='--', color='r', linewidth=1, alpha=.5)
plt.grid(False)

# this isn't quite working. seems what i should be considering is if my hr 
# is just too low. I know my hr shouldn't be 40, unless it's really consistently that way

## really i want the distrib of those that are below the mean. what's that sd
#df_hr_awake_6_to_8.loc[df_hr_awake_6_to_8['hr_vs_smoothed_diff']>0, 'hr_vs_smoothed_diff'] = np.nan 
#
#df_hr_awake_6_to_8['hr_vs_smoothed_diff'].hist(alpha=.5, bins=50)
#plt.grid(False)
#
#df_hr_awake_6_to_8['hr_vs_smoothed_diff'].mean()


#outlier_value = df_hr_awake_6_to_8['hr_vs_smoothed_diff'].mean() - df_hr_awake_6_to_8['hr_vs_smoothed_diff'].std()*2
outlier_value = -10

df_hr_awake_6_to_8['outlier_hr'] = 0
df_hr_awake_6_to_8.loc[(df_hr_awake_6_to_8['hr_vs_smoothed_diff'] < outlier_value), 
                       'outlier_hr'] = 1  # just look at low outliers?
df_hr_awake_6_to_8['outlier_hr'].value_counts(normalize=True)  # 1%. that's a lot
df_hr_awake_6_to_8.loc[df_hr_awake_6_to_8['outlier_hr']==1, 'hr'] = df_hr_awake_6_to_8['hr_rolling_30_min']


# what's my rationale for less than 10 below smoothed line for outliers?
# maybe makes more sense to think about how much a hr point deviated from the prior datapoint?
# look at distribution of these deviations and see what are more thand 2sds below prior
# and elim these? or use the smoothed line, calcu deviations, and take those -2sd below
# the mean deviation. yeah, look at these options rather than just picking one from eyeballing.


df_awake_day = df_hr_awake_6_to_8[df_hr_awake_6_to_8['date']==day]
df_awake_day = df_awake_day[100:400]
plt.plot(df_awake_day.index, df_awake_day['hr'], 
         alpha=.5, color='green', linewidth=1.25)
plt.plot(df_awake_day.index, df_awake_day['hr_rolling_30_min'], 
         alpha=.3, color='grey', linewidth=4)  
plt.grid(axis='y', alpha=.4)

df_awake_day['hr_vs_smoothed_diff'].hist(bins=20)
df_awake_day['hr_vs_smoothed_diff'].min()


# recompute hr rolling 5 or 10 min that i'll get min hr for day using
df_hr_awake_6_to_8['hr_rolling_5min'] = df_hr_awake_6_to_8['hr'].rolling(window='10min', min_periods=10).mean()
df_resting_hr_by_day = df_hr_awake_6_to_8.groupby('date')['hr_rolling_5min'].min().reset_index().rename(columns={'hr_rolling_5min':'resting_hr_5min'})

df_resting_hr_by_day['date'] = pd.to_datetime(df_resting_hr_by_day['date'])
df_resting_hr_by_day['resting_hr_week'] = df_resting_hr_by_day['resting_hr_5min'].rolling(window=7).mean()
df_resting_hr_by_day['resting_hr_month'] = df_resting_hr_by_day['resting_hr_5min'].rolling(window=30).mean()
resting_hr_list = df_resting_hr_by_day['resting_hr_5min'].values
resting_hr_list = np.sort(resting_hr_list)
resting_hr_list = list(resting_hr_list)
resting_hr_list[:10]
resting_hr_list[-10:]
df_resting_hr_by_day = df_resting_hr_by_day.sort_values(by='resting_hr_5min')


# investigate these super low and high ones
day = '2018-09-23'  # can see the low outlier - seems legit
df_awake_day = df_hr_awake_6_to_8[df_hr_awake_6_to_8['date']==day]
#df_awake_day = df_awake_day[100:400]
plt.plot(df_awake_day.index, df_awake_day['hr'], 
         alpha=.5, color='green', linewidth=1.25)
plt.plot(df_awake_day.index, df_awake_day['hr_rolling_30_min'], 
         alpha=.3, color='grey', linewidth=4)  
plt.grid(axis='y', alpha=.4)

df_awake_day[df_awake_day['hour']==9]
df_awake_day[df_awake_day['hour']==10]
df_awake_day[df_awake_day['hour']==11]

day = '2016-11-27'  # can see the low outlier - seems legit
df_awake_day = df_hr_awake_6_to_8[df_hr_awake_6_to_8['date']==day]
#df_awake_day = df_awake_day[100:400]
plt.plot(df_awake_day.index, df_awake_day['hr'], 
         alpha=.5, color='green', linewidth=1.25)
plt.plot(df_awake_day.index, df_awake_day['hr_rolling_30_min'], 
         alpha=.3, color='grey', linewidth=4)  
plt.grid(axis='y', alpha=.4)

df_awake_day[df_awake_day['hour']==12]
df_awake_day[df_awake_day['hour']==13]
df_awake_day[df_awake_day['hour']==14]


# looking at outliers with high resting hr
# not enough data. ok, make some provision here to deal with
# days in which not many datapoints. how? what decision to make

day = '2016-10-09'  # can see the low outlier - seems legit
df_awake_day = df_hr_awake_6_to_8[df_hr_awake_6_to_8['date']==day]
#df_awake_day = df_awake_day[100:400]
plt.plot(df_awake_day.index, df_awake_day['hr'], 
         alpha=.5, color='green', linewidth=1.25)
plt.plot(df_awake_day.index, df_awake_day['hr_rolling_30_min'], 
         alpha=.3, color='grey', linewidth=4)  
plt.grid(axis='y', alpha=.4)

df_awake_day[df_awake_day['hour']==12]
df_awake_day[df_awake_day['hour']==13]
df_awake_day[df_awake_day['hour']==14]


day = '2017-11-12'  # can see the low outlier - seems legit
df_awake_day = df_hr_awake_6_to_8[df_hr_awake_6_to_8['date']==day]
#df_awake_day = df_awake_day[100:400]
plt.plot(df_awake_day.index, df_awake_day['hr'], 
         alpha=.5, color='green', linewidth=1.25)
plt.plot(df_awake_day.index, df_awake_day['hr_rolling_30_min'], 
         alpha=.3, color='grey', linewidth=4)  
plt.grid(axis='y', alpha=.4)

df_awake_day[df_awake_day['hour']==12]
df_awake_day[df_awake_day['hour']==13]
df_awake_day[df_awake_day['hour']==14]

# not enough data. ok, make some provision here to deal with
# days in which not many datapoints. how? what decision to make
# how to tell if a day isn't normal? activity data would help.
# but don't incorporate that now. 
df_hr_awake_6_to_8.groupby('date').size().hist(alpha=.6, bins=20)
plt.axvline(df_hr_awake_6_to_8.groupby('date').size().mean(), linestyle='--')
plt.grid(False)

# filter so only those with 200+ records but why? why trust above that number?
# is this a place for some sort of effect size and power calculation?
# i.e., if i take 200 consecutive datapoints ranomly from those wtih > 500 
# data points how well does a hr-minimum? estimate the actual hr-minium?
# could also do an adjustment towards the mean when fewer and fewer data
# points, or fewer and fewer non-active datapoints (again would need activity
# data for that).
df_datapionts_by_day = df_hr_awake_6_to_8.groupby('date').size().reset_index().rename(columns={0:'size'})
dates_to_keep_list = df_datapionts_by_day[df_datapionts_by_day['size']>200]['date'].values
df_hr_awake_6_to_8 = df_hr_awake_6_to_8[df_hr_awake_6_to_8['date'].isin(dates_to_keep_list)]
len(df_hr_awake_6_to_8['date'].unique())  # 708
len(dates_to_keep_list)


df_resting_hr_by_day = df_resting_hr_by_day[df_resting_hr_by_day['date'].isin(dates_to_keep_list)]

resting_hr_list = df_resting_hr_by_day['resting_hr_5min'].values
resting_hr_list = np.sort(resting_hr_list)
resting_hr_list = list(resting_hr_list)
resting_hr_list[:10]
resting_hr_list[-10:]

df_resting_hr_by_day = df_resting_hr_by_day.sort_values(by='date')


df_resting_hr_by_day['resting_hr_5min'].hist(alpha=.5, bins=25)
plt.grid(False)
plt.axvline(df_resting_hr_by_day['resting_hr_5min'].mean(), linestyle='--')

plt.plot(df_resting_hr_by_day['date'], df_resting_hr_by_day['resting_hr_week'])
plt.plot(df_resting_hr_by_day['date'], df_resting_hr_by_day['resting_hr_month'])

# wow, crazy oscillations
# other ways to plot?
sns.relplot(x='date', y='resting_hr_5min', data=df_resting_hr_by_day, kind='line')  # the ci takes forever
sns.relplot(x='date', y='resting_hr_week', data=df_resting_hr_by_day, kind='line')  # the ci takes forever
g = sns.relplot(x='date', y='resting_hr_month', data=df_resting_hr_by_day, kind='line')  # the ci takes forever
g.fig.autofmt_xdate()


# now try adjusting for time of day and circadian rhythm
df_hr_awake_6_to_8['hr_rolling_5min']

sns.lmplot(x='hour', y='hr_rolling_5min', data=df_hr_awake_6_to_8, 
           scatter_kws={'alpha':.001}, ci=None, order=2)
plt.ylim(55,85)

sns.lmplot(x='hour', y='hr_rolling_5min', data=df_hr_awake_6_to_8, 
           scatter_kws={'alpha':.001}, ci=None, order=1)
plt.ylim(55,85)  # think regular linear order=1 is better than order=2

g = sns.relplot(x='hour', y='hr_rolling_5min', data=df_hr_awake_6_to_8, 
                kind='line', ci=None)  # the ci takes forever
plt.ylim(55,85)

results = smf.ols(formula = 'hr_rolling_5min ~ hour', data=df_hr_awake_6_to_8).fit()
print(results.summary())

df_hr_awake_6_to_8['hr_resid'] = results.resid
df_hr_awake_6_to_8['hr_resid'].hist()

df_hr_awake_6_to_8['hr_resting_adj'] = df_hr_awake_6_to_8['hr_rolling_5min'] + df_hr_awake_6_to_8['hr_resid']

df_hr_awake_6_to_8['hr_resting_adj'].hist(alpha=.5, bins=20)
plt.axvline(df_hr_awake_6_to_8['hr_resting_adj'].mean(), linestyle='--', alpha=.5)
plt.axvline(df_hr_awake_6_to_8['hr_resting_adj'].median(), linestyle='--', alpha=.5)
plt.grid(False)


# compute resting hr again but with this adjusted number
df_hr_awake_6_to_8['hr_rolling_5min_adj'] = df_hr_awake_6_to_8['hr_resting_adj'].rolling(window='10min', min_periods=10).mean()
df_resting_hr_by_day_adj = df_hr_awake_6_to_8.groupby('date')['hr_rolling_5min_adj'].min().reset_index()
df_resting_hr_by_day_adj.rename(columns={'hr_rolling_5min_adj':'resting_hr_adj'}, inplace=True)

df_resting_hr_by_day_adj['resting_hr_adj'].hist(alpha=.6, bins=20)
plt.grid(False)


df_resting_hr_by_day_adj['resting_hr_week'] = df_resting_hr_by_day_adj['resting_hr_adj'].rolling(window=7).mean()
df_resting_hr_by_day_adj['resting_hr_month'] = df_resting_hr_by_day_adj['resting_hr_adj'].rolling(window=30).mean()
resting_hr_list = df_resting_hr_by_day_adj['resting_hr_adj'].values
resting_hr_list = np.sort(resting_hr_list)
resting_hr_list = list(resting_hr_list)
resting_hr_list[:10]
resting_hr_list[-10:]
df_resting_hr_by_day_adj = df_resting_hr_by_day_adj.sort_values(by='resting_hr_adj')


# wow, crazy oscillations
# other ways to plot?
#df_resting_hr_by_day_adj['date'] = pd.to_datetime(df_resting_hr_by_day_adj['date'] )
sns.relplot(x='date', y='resting_hr_adj', data=df_resting_hr_by_day_adj, kind='line')  # the ci takes forever
sns.relplot(x='date', y='resting_hr_week', data=df_resting_hr_by_day_adj, kind='line')  # the ci takes forever
g = sns.relplot(x='date', y='resting_hr_month', data=df_resting_hr_by_day_adj, kind='line')  # the ci takes forever
g.fig.autofmt_xdate()

df_day_hour_hr_min = df_hr_awake_6_to_8.groupby(['date', 'hour'])['hr_rolling_5min'].min().reset_index()
g = sns.relplot(x='hour', y='hr_rolling_5min', data=df_day_hour_hr_min, kind='line')  # the ci takes forever
plt.ylim(50,75)

sns.lmplot(x='hour', y='hr_rolling_5min', data=df_day_hour_hr_min, 
           scatter_kws={'alpha':.001}, order=1)
plt.ylim(50,75)

# skip adjusting for time of day for the time being. if do that, really
# should just get non-active times and compute the relationship between
# hour and hr. or just get the lowest hr each day for each hour and 
# compute the model and residual that way?

df_resting_hr_by_day.tail(15)
df_resting_hr_by_day['resting_hr_5min'].hist(alpha=.5, bins=20)
plt.grid(False)

# use this as resting hr. 
# now can try anys -- what things in sleep are related to a change in resting hr
# though had idea that i'd look at a chage in resting hr adj for the level the initial day
# likely with a cubic function and getting residual from that.

# get prior day hr. then fit model between prior day and today hr. try cubic.
# then get resid as a change score (sorta). and get time of day
df_hr_awake_6_to_8['hr_rolling_5min'] = df_hr_awake_6_to_8['hr_rolling_5min'].astype(float)
df_resting_hr_by_day['resting_hr_5min'] = df_resting_hr_by_day['resting_hr_5min'].astype(float)
df_hr_awake_6_to_8['date'] = pd.to_datetime(df_hr_awake_6_to_8['date'])
df_resting_hr_by_day['date'] = pd.to_datetime(df_resting_hr_by_day['date'])

df_hr_awake_6_to_8['resting_hr_5min'] = df_hr_awake_6_to_8['hr_rolling_5min']

df_resting_hr_merged = pd.merge(df_hr_awake_6_to_8, df_resting_hr_by_day, 
                                on=['date', 'resting_hr_5min'], how='inner')

sns.countplot(x='hour', data=df_resting_hr_merged, color='dodgerblue', alpha=.7)

df_resting_hr_merged.groupby('date').size().value_counts()
df_resting_hr_merged = df_resting_hr_merged.groupby('date').apply(lambda x: x.sample(n=1))
len(df_resting_hr_merged) == len(df_resting_hr_by_day)

df_resting_hr_merged.columns
df_resting_hr_merged = df_resting_hr_merged[['date', 'time', 'hr', 'hour', 
                                             'date_time', 'resting_hr_5min']]

sns.countplot(x='hour', data=df_resting_hr_merged, color='dodgerblue', alpha=.7)
# same pattern as before filtering so only one row per day. most likely to 
# have lowest resting hr during later morning and later afternoon.

df_resting_hr_merged = df_resting_hr_merged.sort_values(by='date')
df_resting_hr_merged['prior_day_rest_hr'] = df_resting_hr_merged['resting_hr_5min'].shift(1)
df_resting_hr_merged['next_day_rest_hr'] = df_resting_hr_merged['resting_hr_5min'].shift(-1)

df_resting_hr_merged.head()

df_resting_hr_merged['prior_record_days_diff'] = (df_resting_hr_merged['date'] - df_resting_hr_merged['date'].shift(1)) / np.timedelta64(1, 'D') 
df_resting_hr_merged.loc[df_resting_hr_merged['prior_record_days_diff']!=1, 'prior_day_rest_hr'] = np.nan
df_resting_hr_merged[['prior_day_rest_hr', 'prior_record_days_diff']]

sns.relplot(x='prior_day_rest_hr', y='resting_hr_5min', data=df_resting_hr_merged, alpha=.5)

sns.lmplot(x='prior_day_rest_hr', y='resting_hr_5min', data=df_resting_hr_merged,
           scatter_kws={'alpha':.1}, order=3)


df_filtered = df_resting_hr_merged[(df_resting_hr_merged['resting_hr_5min']>49) & (df_resting_hr_merged['resting_hr_5min']<56) ]

results = smf.ols(formula = 'resting_hr_5min ~ prior_day_rest_hr', data=df_filtered).fit()
print(results.summary())

results = smf.ols(formula = 'resting_hr_5min ~ prior_day_rest_hr + I(prior_day_rest_hr**2) + I(prior_day_rest_hr**3)', data=df_filtered).fit()
print(results.summary())

sns.lmplot(x='prior_day_rest_hr', y='resting_hr_5min', data=df_filtered,
           scatter_kws={'alpha':.1}, order=3)

sns.lmplot(x='prior_day_rest_hr', y='resting_hr_5min', data=df_filtered,
           scatter_kws={'alpha':.1}, order=1)


df_resting_hr_merged['resting_hr_5min'].hist()
df_filtered['resting_hr_5min'].hist()

results = smf.ols(formula = 'resting_hr_5min ~ prior_day_rest_hr + I(prior_day_rest_hr**2) + I(prior_day_rest_hr**3)', data=df_resting_hr_merged).fit()
print(results.summary())

df_resting_hr_merged['hr_rest_change_resid'] = results.resid
df_resting_hr_merged['hr_rest_change'] = df_resting_hr_merged['resting_hr_5min'] - df_resting_hr_merged['prior_day_rest_hr']

sns.lmplot(x='hr_rest_change_resid', y='hr_rest_change', data=df_resting_hr_merged,
           scatter_kws={'alpha':.1}, order=3)

# showing that no relationship between prior resting hr and the resid change metric
sns.lmplot(x='prior_day_rest_hr', y='hr_rest_change_resid', data=df_resting_hr_merged,
           scatter_kws={'alpha':.1}, order=3)
# showing that ther is a relationship between the prior resting hr and the regular change metric
# such that the lower the prior hr, the more likely there is to be an increas in hr from day to day
# and the higher the prior hr, the more likely there is to be a decrease.
sns.lmplot(x='prior_day_rest_hr', y='hr_rest_change', data=df_resting_hr_merged,
           scatter_kws={'alpha':.1}, order=3)

df_resting_hr_merged[['prior_day_rest_hr', 'resting_hr_5min']].corr()

df_fitbit_measure = pd.read_excel('resting_hr_fitbit_measure.xlsx')
date_to_fitbit_resting_hr_measure = dict(zip(df_fitbit_measure['date'], 
                                         df_fitbit_measure['resting_hr_fb_measure']))    

df_resting_hr_merged['fibit_resting_hr_measure'] = df_resting_hr_merged['date'].map(date_to_fitbit_resting_hr_measure)

plt.scatter(df_resting_hr_merged['resting_hr_5min'], 
            df_resting_hr_merged['fibit_resting_hr_measure'],
            alpha=.5)

df_resting_hr_merged[['prior_day_rest_hr', 'resting_hr_5min', 'next_day_rest_hr',
                      'fibit_resting_hr_measure']].corr()

df_resting_hr_merged[['prior_day_rest_hr', 'resting_hr_5min', 'next_day_rest_hr',
                      'fibit_resting_hr_measure']].mean()

# strange that the fitbit measure of resting hr corr better with prior day's 
# resting hr than the actual day. but I checked and almost positive the dates 
# are correct. makes me think that fitbit is using prior days hr to help 
# compute current day's hr? would that make sense? or it's some moving/smoothed 
# avg of yesterday's hr and todays hr? i could email them. email a data scientist
# who works there?


plt.scatter(df_resting_hr_merged['prior_day_rest_hr'], 
            df_resting_hr_merged['fibit_resting_hr_measure'],
            alpha=.5)

sns.lmplot(x='prior_day_rest_hr', y='fibit_resting_hr_measure', 
           data=df_resting_hr_merged)

df_resting_hr_merged['resting_prior_and_today_hr'] = df_resting_hr_merged[['prior_day_rest_hr', 'resting_hr_5min']].mean(axis=1)

df_resting_hr_merged[['resting_prior_and_today_hr', 'fibit_resting_hr_measure']].corr()

plt.scatter(df_resting_hr_merged['resting_prior_and_today_hr'], 
            df_resting_hr_merged['fibit_resting_hr_measure'],
            alpha=.5)

sns.lmplot(x='resting_prior_and_today_hr', y='fibit_resting_hr_measure', 
           data=df_resting_hr_merged)


# LEFT OFF HERE
# conrol for hour too? not sure.
# but why don't i match this resid hr change metric up with sleep metrics
# other ways to measure resting hr?
# record resting hr by hand from fitbit dash for a year to see how it
# matches up with my calculation of resting hr.

# way to show that hr during sleep isn't like hr normally?


















































# what time of day am i seeing the minimum hr? esp the ones at the extremes.
# the extremes are suspicious. esp the super low ones. might suggest outlier
# values that i should remove using the smoothing technique in the jup nb.
# adjust for time of day. using quadratic model?
# set time between 9am and 8pm for getting min hr?

#sns.boxplot(x='hour', y='hr', data=df_hr, color='dodgerblue')

sns.barplot(df_hr['date_time'].dt.hour, df_hr['hr'], 
            data=df_hr, alpha=.5, color='dodgerblue', ci=None)
plt.ylim(30,75)
sns.despine()
# plotting just times with lower hr
#df_non_active = df_hr[df_hr['hr']<=80]
#sns.barplot(df_non_active['date_time'].dt.hour, df_non_active['hr'], 
#            data=df_non_active, alpha=.5, color='dodgerblue', ci=None)
#plt.ylim(30,75)
#sns.despine()

# i think these oscillations are from activity. even if i filter df to
# times where hr > 70 or 80, oscillations are there. i think because 
# even though i'm likely not active at those moments, i'm more active
# during those times and so more often recoving from high hr when i'm 
# not active. suggests my circadian hr rhythm is to increase througout 
# the day til around 8pm and then decrease til around 5am, then increase again.


sns.lmplot(x='hour', y='hr', data=df_hr, ci=None, scatter_kws={'alpha':.005}, order=3)

sns.lmplot(x='hour', y='hr', data=df_hr, ci=None, scatter_kws={'alpha':.001}, order=3)
plt.ylim(40,80)

# lowess doesn't work well here
#sns.lmplot(x='hour', y='hr', data=df_hr, ci=None, scatter_kws={'alpha':.0}, lowess=True)
#plt.ylim(45,75)

# compute resting hr 
# 5-10 min moving avg and take min
# but adjust for time of day? 
# could take the min 5-10 min moving avg from 8-6
# and then adjust for the time. 
# if i just plot 8-6 is that linear?
sns.barplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], 
                                         alpha=.5, color='dodgerblue', ci=None)
plt.ylim(40,80)
sns.despine()

g = sns.relplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], 
                                             kind='line', ci=None)  # the ci takes forever
plt.ylim(40,80)
plt.xlim(5,21)



sns.lmplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], 
                                        ci=None, scatter_kws={'alpha':.001}, order=2)
plt.ylim(40,80)
plt.xlim(5,21)

sns.relplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], 
                                         kind='line', ci=None)  # the ci takes forever

# this might be pretty good to get adjusted hr resting, i.e., adjust for 
# time af day and circadian rhythm. overlay this with the actual lineplot
fig, ax = plt.subplots()
sns.regplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], 
                                        ci=None, scatter_kws={'alpha':.001}, 
                                        order=2, ax=ax)
sns.relplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], 
                                         kind='line', ci=None, ax=ax2)  # the ci takes forever
ax2 = ax.twinx()
sns.plt.show()

sns.regplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], order=2)


fig, ax = plt.subplots()
sns.regplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], 
                                         order=2, ci=None, ax=ax, scatter_kws={'alpha':0})
sns.relplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], 
                                         kind='line', ci=None, ax=ax)
plt.ylim(40,80)



#sns.lmplot(x='hour', y='hr', data=df_hr[(df_hr['hour']>=6) & (df_hr['hour']<=20)], 
#                                        ci=None, scatter_kws={'alpha':.002}, order=3)
#plt.ylim(40,80)
#plt.xlim(5,21)



hour_list = list(range(0,26,2))
hour_string_list = [str(hour) for hour in hour_list]
plt.xticks(hour_list, hour_string_list, fontsize=10)


# =========
# LEFT OFF
# =========




# -------------
# activity data

# when up-pickle file, just focus on summary stats for ea day
len(stats_pickled_file['summary'])
stats_pickled_file['summary'].keys()
stats_pickled_file['summary']['distance']
stats_pickled_file['summary']['calories']
stats_pickled_file['summary']['activityLevels']
stats_pickled_file['summary']['elevation']
stats_pickled_file['summary']['steps']
stats_pickled_file['summary']['floors']
stats_pickled_file['summary']['heartRateZones']






# -----------
# get contious hr
# set date range for getting data here:
dates = pd.date_range('10/01/2016', '02/03/17', freq='D')
dates = pd.date_range('02/01/2017', '05/01/17', freq='D')  # retreived on 5/3/17
dates = pd.date_range('05/01/2017', '08/10/17', freq='D')  # retreived on 8/12/17
dates = pd.date_range('08/10/2017', '02/01/18', freq='D')  # retreived on 7/6/18
dates = pd.date_range('02/02/2018', '07/29/18', freq='D')  # ...still need to get this on 7/7/18


dates = pd.date_range('08/10/2017', '08/015/2017', freq='D')


def get_hr_continuous(authd_client, dates, time_slice):
    #continuous_hr_day = {'date':[], 'time':[], 'hr':[]}
    continuous_min_hr_day = {'date':[], 'time':[], 'hr':[]}
    for date in dates[:]:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        stats_hr = authd_client.intraday_time_series('activities/heart', base_date=date, detail_level='1min', start_time='00:00', end_time='23:59')
        hr_data = stats_hr['activities-heart-intraday']['dataset']
        for i in range(len(hr_data)):
            #print(i)
            time = hr_data[i]['time']
            hr = hr_data[i]['value']
            continuous_min_hr_day['date'].append(date)
            continuous_min_hr_day['time'].append(time)
            continuous_min_hr_day['hr'].append(hr)
    return continuous_min_hr_day

continuous_min_hr_day = get_hr_continuous(authd_client, dates, '1min')
# wait an hr

#dt_hr_continuous = pd.DataFrame(continuous_hr_day)
#dt_hr_continuous.head(25)
#dt_hr_continuous.tail(25)
#len(dt_hr_continuous)
#dt_hr_continuous.to_csv('dt_hr_continuous_2017_02_03')

dt_hr_continuous_min = pd.DataFrame(continuous_min_hr_day)
dt_hr_continuous_min.head(25)
dt_hr_continuous_min.tail(25)
#dt_hr_continuous_min.to_csv('df_hr_continuous_min_2017_02_01_to_2017_05_01.csv')
#dt_hr_continuous_min.to_csv('df_hr_continuous_min_2017_05_01_to_2017_08_10.csv')
#dt_hr_continuous_min.to_csv('df_hr_continuous_min_2017_08_10_to_2018_02_01.csv')


#dt_hr_continuous_min = pd.read_csv('dt_hr_continuous_min_2017_02_03')
#dt_hr_continuous_min.head()
#del dt_hr_continuous_min['Unnamed: 0']
dt_hr_continuous_min['date'] = pd.to_datetime(dt_hr_continuous_min['date'])
dt_hr_continuous_min['date_time'] = pd.to_datetime(dt_hr_continuous_min['date'].astype(str) + ' ' + dt_hr_continuous_min['time'])
dt_hr_continuous_min['hour'] = dt_hr_continuous_min['date_time'].dt.hour

df_night = dt_hr_continuous_min[dt_hr_continuous_min['hour']<8]
#dt_hr_continuous_min['hr_moving_avg'] = pd.rolling_mean(dt_hr_continuous_min['hr'], window=100)
df_night['hr_moving_avg'] = df_night['hr'].rolling(window=4000).mean()
plt.plot(df_night['date_time'], df_night['hr_moving_avg'])
plt.xticks(rotation=50)

#df_night_jan = df_night[df_night['date']>'2017-01-01']
#df_night_jan['hr_moving_avg'] = df_night_jan['hr'].rolling(window=1000).mean()
#plt.plot(df_night_jan['date_time'], df_night_jan['hr_moving_avg'])
#plt.xticks(rotation=50)


# --------------
# get resting hr
#dates = pd.date_range('02/01/2017', '08/10/17', freq='D')

dates[89]

def get_resting_hr(authd_client, dates, start, end):
    resting_hr_day = {'date':[], 'date_2':[], 'hr_resting':[]}
    for date in dates[start:end]:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        stats_resting_hr = authd_client.time_series('activities/heart', base_date=date, period='1d')
        date_2 = stats_resting_hr['activities-heart'][0]['dateTime']
        hr_resting = stats_resting_hr['activities-heart'][0]['value']['restingHeartRate']
        resting_hr_day['date'].append(date)
        resting_hr_day['date_2'].append(date_2)
        resting_hr_day['hr_resting'].append(hr_resting)
    return resting_hr_day

# these dates started at 5-1-2017
resting_hr = get_resting_hr(authd_client, dates, 0, None)
resting_hr_5_1_to_8_10_2017 = copy.deepcopy(resting_hr)
df_resting_hr_5_1_to_8_10_2017 = pd.DataFrame(resting_hr_5_1_to_8_10_2017)

# these dates started at 2-1-2017
resting_hr = get_resting_hr(authd_client, dates, 0, 49)
resting_hr_2_1_to_3_21_2017 = copy.deepcopy(resting_hr)
df_resting_hr_2_1_to_3_21_2017 = pd.DataFrame(resting_hr_2_1_to_3_21_2017)

resting_hr = get_resting_hr(authd_client, dates, 50, 89)
resting_hr_3_23_to_4_30_2017 = copy.deepcopy(resting_hr)
df_resting_hr_3_23_to_4_30_2017 = pd.DataFrame(resting_hr_3_23_to_4_30_2017)

df_resting = pd.concat([df_resting_hr_2_1_to_3_21_2017,
                                 df_resting_hr_3_23_to_4_30_2017],
                                 ignore_index=True)

#df_resting.to_csv('df_resting_hr_2_1_to_4_40_2017.csv')
#df_resting_hr.to_csv('df_resting_hr_2017_02_03.csv')
#df_resting_hr_5_1_to_8_10_2017.to_csv('df_resting_hr_5_1_to_8_10_2017.csv')


# dates = pd.date_range('08/10/2017', '02/01/18', freq='D')
resting_hr = get_resting_hr(authd_client, dates, 0, 74)
df_resting_hr_2017_08_10_to_2017_10_22 = pd.DataFrame(resting_hr)

resting_hr = get_resting_hr(authd_client, dates, 74, 75)
df_resting_hr_2017_10_23_to_2017_10_23 = pd.DataFrame(resting_hr)

resting_hr = get_resting_hr(authd_client, dates, 76, 90)
df_resting_hr_2017_10_25_to_2017_11_7 = pd.DataFrame(resting_hr)

resting_hr = get_resting_hr(authd_client, dates, 90, 98)
df_resting_hr_2017_11_8_to_2017_11_11 = pd.DataFrame(resting_hr)

resting_hr = get_resting_hr(authd_client, dates, 99, 125)
df_resting_hr_2017_11_17_to_2017_12_12 = pd.DataFrame(resting_hr)

resting_hr = get_resting_hr(authd_client, dates, 125, 133)
df_resting_hr_2017_12_13_to_2017_12_20 = pd.DataFrame(resting_hr)

resting_hr = get_resting_hr(authd_client, dates, 133, 159)
df_resting_hr_2017_12_21_to_2018_1_15 = pd.DataFrame(resting_hr)

resting_hr = get_resting_hr(authd_client, dates, 161, -1)
df_resting_hr_2018_1_18_to_2018_1_31 = pd.DataFrame(resting_hr)


df_resting = pd.concat([df_resting_hr_2017_08_10_to_2017_10_22,
                        df_resting_hr_2017_10_23_to_2017_10_23,
                        df_resting_hr_2017_10_25_to_2017_11_7,
                        df_resting_hr_2017_11_8_to_2017_11_11,
                        df_resting_hr_2017_11_17_to_2017_12_12,
                        df_resting_hr_2017_12_13_to_2017_12_20,
                        df_resting_hr_2017_12_21_to_2018_1_15,
                        df_resting_hr_2018_1_18_to_2018_1_31],
                        ignore_index=True)

df_resting.to_csv('df_resting_hr_2017_08_10_to_2018_01_31.csv')

dates


df_resting['date'] = pd.to_datetime(df_resting['date'])
df_resting['hr_resting_moving_avg'] = df_resting['hr_resting'].rolling(window=10).mean()
plt.plot(df_resting['date'], df_resting['hr_resting_moving_avg'])
plt.xticks(rotation=50)


# --------------
# get sleep data
# is it possible this has sleep cycle info?
#date = dates[10]

# explore -- sleep stages?
stats_sleep = authd_client.sleep(date=date)

stats_sleep['data']

sleep_by_min_data = stats_sleep['sleep'][0]['minuteData']  # min by min whether asleep or not


dir(authd_client)
authd_client.RESOURCE_LIST
authd_client.heart(ARGUMENT?)
authd_client.intraday_time_series(ARGUMENT?)
authd_client.sleep(date=date)
authd_client.time_series(ARGUMENT?)

authd_client.time_series('sleep', base_date=date, period='1d')
x = authd_client.time_series('activities/heart', base_date=date, period='1d')
x['activities-heart-intraday']['dataset'][:30]

x = authd_client.time_series('sleep/sleep_log', base_date=date, period='1d')


authd_client.get_sleep(date=pd.to_datetime(date))


dir(fitbit)

x = authd_client.sleep(date=date)
x['sleep'][0]['isMainSleep']
x['sleep'][0]['minuteData']
len(x['sleep'])
x['sleep'][0]['minuteData']

authd_client.intraday_time_series('sleep', base_date=date)
dir(authd_client.intraday_time_series)


uri = "{0}/{1}/user/-/activities/heart/date/{date}/{end_date}/{detail_level}.json"
end_date = date

url = uri.format(consumer_key),
    date=date,
    end_date=end_date,
    detail_level='1min'
)


x = '{0}{4}hello'
x.format('5','h')

url = 'https://api.fitbit.com/1.2/user/2286HS/sleep/date/2018-06-01.json'
make_request(url)


import requests
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import MobileApplicationClient

client_id = consumer_key
scope = ["activity", "heartrate", "location", "nutrition", "profile", "settings",
"sleep", "social", "weight"]

client = MobileApplicationClient(client_id)
fitbit = OAuth2Session(client_id, scope=scope)
authorization_url = "https://www.fitbit.com/oauth2/authorize"
auth_url, state = fitbit.authorization_url(authorization_url)
print("Visit this page in your browser: {}".format(auth_url))
token
r = fitbit.get('https://api.fitbit.com/1/user/-/sleep/goal.json')
r = fitbit.get('https://api.fitbit.com/1.2/user/-/sleep/2018-06-01.json')


sleep_stage_datapoint = sleep_continuous_stages[0]

date = pd.to_datetime('2017-05-01')


	with open('MoviesByMonth2011.pkl', 'wb') as picklefile:
		pickle.dump(d1, picklefile)
	with open('oneMonthMovies.pkl', 'rb') as picklefile:
		d2 = pickle.load(picklefile)


# has stages:
# stages data starts about '2017-06-15'
date = pd.to_datetime('2017-06-15')

date = pd.to_datetime('2017-07-01')

date = pd.to_datetime('2017-09-01')

date = pd.to_datetime('2017-11-01')

date = pd.to_datetime('2017-12-01')

date = pd.to_datetime('2018-02-01')

date = pd.to_datetime('2018-04-01')
# has two periods of sleep

date = pd.to_datetime('2018-05-01')

# doesn't have timing of stages
date = pd.to_datetime('2018-06-01')
date = pd.to_datetime('2018-06-02')

   'type': 'classic'}],
   'type': 'stages'}],



# test date range
#dates = pd.date_range('06/15/2017', '06/18/2017', freq='D')

# new f for API 1.2
#start=0
#end=None
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

pickle_sleep_continuous(authd_client, dates, 0, None)

# full date range
dates = pd.date_range('06/18/2017', '07/31/2018', freq='D')
pickle_sleep_continuous(authd_client, dates, 0, None)


date = dates[164]
date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')

with open('sleep'+date+'.pkl', 'rb') as picklefile:
    sleep_test_data = pickle.load(picklefile)

dates = pd.date_range('11/29/2017', '07/31/2018', freq='D')
pickle_sleep_continuous(authd_client, dates, 0, None)

dates = pd.date_range('05/22/2018', '08/10/2018', freq='D')
pickle_sleep_continuous(authd_client, dates, 0, None)

# can see in folder that stopped at 2017-11-19.
# it's because too many requests. so need to wait 1 hr.
# i got about 6 mo of data with one shot. ok.


# code for importing and manipulating sleep data (put below)
# test date range
dates = pd.date_range('06/18/2017', '06/30/2017', freq='D')

date = dates[0]
date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')

with open('sleep'+date+'.pkl', 'rb') as picklefile:
    sleep_test_data = pickle.load(picklefile)

if sleep_test_data['sleep'][0]['type'] == 'classic':
    sleep_wakings = sleep_test_data['sleep'][0]['levels']['data']
    duration = sleep_test_data['sleep'][0]['duration']
    efficiency = sleep_test_data['sleep'][0]['efficiency']
    end_time = sleep_test_data['sleep'][0]['endTime']
    main_sleep = sleep_test_data['sleep'][0]['isMainSleep']
    summary = sleep_test_data['sleep'][0]['levels']['summary']
elif sleep_test_data['sleep'][0]['type'] == 'stages':
    sleep_stages = sleep_test_data['sleep'][0]['levels']['data']
    sleep_wakings = sleep_test_data['sleep'][0]['levels']['shortData']


df_sleep_stages = pd.DataFrame(sleep_stages)
df_sleep_stages.dtypes

df_sleep_stages['date_time'] = pd.to_datetime(df_sleep_stages['dateTime'])
df_sleep_stages['date'] = pd.to_datetime(df_sleep_stages['date_time'].dt.date)
df_sleep_stages['time'] = df_sleep_stages['date_time'].dt.time
df_sleep_stages.set_index('date_time', inplace=True)
del df_sleep_stages['dateTime']

# plan: turn time part into just hour and minute. cut off sec.
# then create dict and map onto a resampled-by-min df
df_sleep_stages['time'] = df_sleep_stages['time'].astype(str)
#df_sleep_stages['time'].astype(str).str.split(':')
df_sleep_stages['hour_min'] = df_sleep_stages['time'].astype(str).str[:5]
df_sleep_stages['hour_min'] = df_sleep_stages['hour_min']+':00'
# can now use this to map sleep stages onto a df resampled by min

df_sleep_stages['date_time'] = pd.to_datetime(df_sleep_stages['date'].astype(str) + ' ' + df_sleep_stages['hour_min'])
datetime_to_sleep_stage_dict = dict(zip(df_sleep_stages['date_time'], df_sleep_stages['level']))

#df_sleep_stages.resample('min').interpolate()
df_sleep_stages = df_sleep_stages.resample('min').median()
len(df_sleep_stages)
df_sleep_stages['date_time'] = df_sleep_stages.index
df_sleep_stages['sleep_stage'] = df_sleep_stages['date_time'].map(datetime_to_sleep_stage_dict)








stats_sleep = authd_client.sleep(date=date)

    sleep_stages = {'date':[], 'time':[], 'sleep_stage':[], 'seconds_in_stage':[]}
    sleep_awake = {'date':[], 'time':[], 'sleep_awake':[], 'seconds_awake':[]}

        if stats_sleep['sleep'][0]['type'] == 'classic':
            sleep_wakings = stats_sleep['sleep'][0]['levels']['data']
            duration = stats_sleep['sleep'][0]['duration']
            efficiency = stats_sleep['sleep'][0]['efficiency']
            end_time = stats_sleep['sleep'][0]['endTime']
            main_sleep = stats_sleep['sleep'][0]['isMainSleep']
            summary = stats_sleep['sleep'][0]['levels']['summary']

        if stats_sleep['sleep'][0]['type'] == 'stages':
            sleep_stages = stats_sleep['sleep'][0]['levels']['data']
            sleep_wakings = stats_sleep['sleep'][0]['levels']['shortData']


        # incorporate loop when i slept more than once in a day?
        len(stats_sleep)  # sleep and summary
        len(stats_sleep['sleep'])  # 2 if two sleeps
        for i in range(0, len(stats_sleep['sleep'])):
            print(i)
            if stats_sleep['sleep'][i]['isMainSleep']:
                sleep_stretch = i
            else:
                None


        stats_sleep['summary']


        sleep_continuous_stages = stats_sleep['sleep'][0]['levels']['data']  # min by min whether asleep or not
        sleep_continuous_waking = stats_sleep['sleep'][0]['levels']['shortData']  # min by min whether asleep or not
        for sleep_stage_datapoint in sleep_continuous_stages:
            sleep_stages['date'].append(date)
            sleep_stages['sleep_stage'].append(sleep_stage_datapoint['level'])
            sleep_stages['seconds_in_stage'].append(sleep_stage_datapoint['seconds'])
            sleep_stages['time'].append(sleep_stage_datapoint['dateTime'])
        for sleep_awake_datapoint in sleep_continuous_waking:
            sleep_awake['date'].append(date)
            sleep_awake['sleep_awake'].append(sleep_awake_datapoint['level'])
            sleep_awake['seconds_awake'].append(sleep_awake_datapoint['seconds'])
            sleep_awake['time'].append(sleep_awake_datapoint['dateTime'])
     return sleep_stages, sleep_awake





def get_sleep_continuous(authd_client, dates, start, end):
    sleep_stages = {'date':[], 'time':[], 'sleep_stage':[], 'seconds_in_stage':[]}
    sleep_awake = {'date':[], 'time':[], 'sleep_awake':[], 'seconds_awake':[]}
    for date in dates[start:end]:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        stats_sleep = authd_client.sleep(date=date)

        if stats_sleep['sleep'][0]['type'] == 'classic':
            sleep_wakings = stats_sleep['sleep'][0]['levels']['data']
            duration = stats_sleep['sleep'][0]['duration']
            efficiency = stats_sleep['sleep'][0]['efficiency']
            end_time = stats_sleep['sleep'][0]['endTime']
            main_sleep = stats_sleep['sleep'][0]['isMainSleep']
            summary = stats_sleep['sleep'][0]['levels']['summary']

        if stats_sleep['sleep'][0]['type'] == 'stages':
            sleep_stages = stats_sleep['sleep'][0]['levels']['data']
            sleep_wakings = stats_sleep['sleep'][0]['levels']['shortData']


        # incorporate loop when i slept more than once in a day?
        len(stats_sleep)  # sleep and summary
        len(stats_sleep['sleep'])  # 2 if two sleeps
        for i in range(0, len(stats_sleep['sleep'])):
            print(i)
            if stats_sleep['sleep'][i]['isMainSleep']:
                sleep_stretch = i
            else:
                None


        stats_sleep['summary']


        sleep_continuous_stages = stats_sleep['sleep'][0]['levels']['data']  # min by min whether asleep or not
        sleep_continuous_waking = stats_sleep['sleep'][0]['levels']['shortData']  # min by min whether asleep or not
        for sleep_stage_datapoint in sleep_continuous_stages:
            sleep_stages['date'].append(date)
            sleep_stages['sleep_stage'].append(sleep_stage_datapoint['level'])
            sleep_stages['seconds_in_stage'].append(sleep_stage_datapoint['seconds'])
            sleep_stages['time'].append(sleep_stage_datapoint['dateTime'])
        for sleep_awake_datapoint in sleep_continuous_waking:
            sleep_awake['date'].append(date)
            sleep_awake['sleep_awake'].append(sleep_awake_datapoint['level'])
            sleep_awake['seconds_awake'].append(sleep_awake_datapoint['seconds'])
            sleep_awake['time'].append(sleep_awake_datapoint['dateTime'])
     return sleep_stages, sleep_awake




df_sleep_awake = pd.DataFrame(sleep_awake)
df_sleep_stages = pd.DataFrame(sleep_stages)
df_sleep_stages.dtypes


# old f for API 1
def get_sleep_continuous(authd_client, dates, start, end):
    sleep_day = {'date':[], 'time':[], 'sleep_status':[]}
    for date in dates[start:end]:
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        stats_sleep = authd_client.sleep(date=date)
        sleep_by_min_data = stats_sleep['sleep'][0]['minuteData']  # min by min whether asleep or not
        for i in range(len(sleep_by_min_data)):
            time = sleep_by_min_data[i]['dateTime']
            sleep_status = sleep_by_min_data[i]['value']
            sleep_day['date'].append(date)
            sleep_day['time'].append(time)
            sleep_day['sleep_status'].append(sleep_status)
    return sleep_day

sleep_days = get_sleep_continuous(authd_client, dates, 0, 3)


sleep_days = get_sleep_continuous(authd_client, dates, 0, 2)
sleep_days_5_1_to_5_2_2017 = copy.deepcopy(sleep_days)
df_sleep_days_5_1_to_5_2_2017 = pd.DataFrame(sleep_days_5_1_to_5_2_2017)

sleep_days = get_sleep_continuous(authd_client, dates, 3, 12)
sleep_days_5_4_to_5_12_2017 = copy.deepcopy(sleep_days)
df_sleep_days_5_4_to_5_12_2017 = pd.DataFrame(sleep_days_5_4_to_5_12_2017)

sleep_days = get_sleep_continuous(authd_client, dates, 12, 13)
sleep_days_5_13_to_5_13_2017 = copy.deepcopy(sleep_days)
df_sleep_days_5_13_to_5_13_2017 = pd.DataFrame(sleep_days_5_13_to_5_13_2017)

sleep_days = get_sleep_continuous(authd_client, dates, 15, None)
sleep_days_5_16_to_8_10_2017 = copy.deepcopy(sleep_days)
df_sleep_days_5_16_to_8_10_2017 = pd.DataFrame(sleep_days_5_16_to_8_10_2017)

# if cuts me off, need to wait and hour, or til end of hour?

# 1 ("asleep"), 2 ("restless"), or 3 ("awake")

df_sleep_continuous = pd.concat([df_sleep_days_5_1_to_5_2_2017,
                                 df_sleep_days_5_4_to_5_12_2017,
                                 df_sleep_days_5_13_to_5_13_2017,
                                 df_sleep_days_5_16_to_8_10_2017],
                                 ignore_index=True)

df_sleep_continuous.head(10)
df_sleep_continuous.tail(10)
#df_sleep_continuous.to_csv('df_sleep_continuous_2017_05_01_to_2017_08_10.csv')


# for dates pd.date_range('08/10/2017', '02/01/18', freq='D')
len(dates)  # 176
dates[74]
sleep_days = get_sleep_continuous(authd_client, dates, 0, 74)
df_sleep_days_2017_08_10_to_2017_10_22 = pd.DataFrame(sleep_days)

sleep_days = get_sleep_continuous(authd_client, dates, 74, 75)
df_sleep_days_2017_10_23_to_2017_10_23 = pd.DataFrame(sleep_days)

sleep_days = get_sleep_continuous(authd_client, dates, 76, 125)
df_sleep_days_2017_10_25_to_2017_12_12 = pd.DataFrame(sleep_days)

sleep_days = get_sleep_continuous(authd_client, dates, 125, 133)
df_sleep_days_2017_12_13_to_2017_12_20 = pd.DataFrame(sleep_days)

sleep_days = get_sleep_continuous(authd_client, dates, 134, -1)
df_sleep_days_2017_12_22_to_2018_1_31 = pd.DataFrame(sleep_days)

# then concat and save
df_sleep_continuous = pd.concat([df_sleep_days_2017_08_10_to_2017_10_22,
                                 df_sleep_days_2017_10_23_to_2017_10_23,
                                 df_sleep_days_2017_10_25_to_2017_12_12,
                                 df_sleep_days_2017_12_13_to_2017_12_20,
                                 df_sleep_days_2017_12_22_to_2018_1_31],
                                 ignore_index=True)

df_sleep_continuous.head(10)
df_sleep_continuous.tail(10)
#df_sleep_continuous.to_csv('df_sleep_continuous_2017_08_10_to_2018_01_31.csv')






# -----------------
# get sleep summary stats for ea day
# can i get sleep cycles here now?

# test (and test a year ago for sleep cycles too)
dates = pd.date_range('07/01/2018', '07/05/18', freq='D')

#for date in dates[start:end]:
date = dates[1]
    date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
    stats_sleep = authd_client.sleep(date=date)

    deep_sleep = stats_sleep['summary']['stages']['deep']
    rem_sleep = stats_sleep['summary']['stages']['rem']
    light_sleep = stats_sleep['summary']['stages']['light']
    wake_during_sleep = stats_sleep['summary']['stages']['wake']
    # add this to f below


def get_sleep_summary(authd_client, dates, start, end):
    sleep_summary_day = {'awake_count':[], 'awake_duration':[], 'awakenings_count':[],
                         'date_of_sleep':[], 'duration':[], 'efficiency':[], 'is_main_sleep':[],
                        'minutes_asleep':[], 'minutes_awake':[], 'minutes_to_fall_asleep':[],
                        'restless_count':[], 'restless_duration':[], 'total_minutes_asleep':[],
                        'total_sleep_records':[], 'total_time_in_bed':[]}
    for date in dates[start:end]:
    #date = dates[1]
        date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
        print(date)
        stats_sleep = authd_client.sleep(date=date)
        # get specific data points
        awake_count = stats_sleep['sleep'][0]['awakeCount']
        awake_duration = stats_sleep['sleep'][0]['awakeDuration']
        awakenings_count = stats_sleep['sleep'][0]['awakeningsCount']
        date_of_sleep = stats_sleep['sleep'][0]['dateOfSleep']
        duration = stats_sleep['sleep'][0]['duration']
        efficiency = stats_sleep['sleep'][0]['efficiency']
        is_main_sleep = stats_sleep['sleep'][0]['isMainSleep']
        minutes_asleep = stats_sleep['sleep'][0]['minutesAsleep']
        minutes_awake = stats_sleep['sleep'][0]['minutesAwake']
        minutes_to_fall_asleep = stats_sleep['sleep'][0]['minutesToFallAsleep']
        restless_count = stats_sleep['sleep'][0]['restlessCount']
        restless_duration = stats_sleep['sleep'][0]['restlessDuration']
        total_minutes_asleep = stats_sleep['summary']['totalMinutesAsleep']
        total_sleep_records = stats_sleep['summary']['totalSleepRecords']
        total_time_in_bed = stats_sleep['summary']['totalTimeInBed']
        # append data points to dict
        sleep_summary_day['awake_count'].append(awake_count)
        sleep_summary_day['awake_duration'].append(awake_duration)
        sleep_summary_day['awakenings_count'].append(awakenings_count)
        sleep_summary_day['date_of_sleep'].append(date_of_sleep)
        sleep_summary_day['duration'].append(duration)
        sleep_summary_day['efficiency'].append(efficiency)
        sleep_summary_day['is_main_sleep'].append(is_main_sleep)
        sleep_summary_day['minutes_asleep'].append(minutes_asleep)
        sleep_summary_day['minutes_awake'].append(minutes_awake)
        sleep_summary_day['minutes_to_fall_asleep'].append(minutes_to_fall_asleep)
        sleep_summary_day['restless_count'].append(restless_count)
        sleep_summary_day['restless_duration'].append(restless_duration)
        sleep_summary_day['total_minutes_asleep'].append(total_minutes_asleep)
        sleep_summary_day['total_sleep_records'].append(total_sleep_records)
        sleep_summary_day['total_time_in_bed'].append(total_time_in_bed)
    return sleep_summary_day


sleep_summary = get_sleep_summary(authd_client, dates, 0, 2)
sleep_summary_5_1_to_5_2_2017 = copy.deepcopy(sleep_summary)
df_sleep_summary_5_1_to_5_2_2017 = pd.DataFrame(sleep_summary_5_1_to_5_2_2017)

sleep_summary = get_sleep_summary(authd_client, dates, 3, 13)
sleep_summary_5_4_to_5_13_2017 = copy.deepcopy(sleep_summary)
df_sleep_summary_5_4_to_5_13_2017 = pd.DataFrame(sleep_summary_5_4_to_5_13_2017)

sleep_summary = get_sleep_summary(authd_client, dates, 15, None)
sleep_summary_5_16_to_8_10_2017 = copy.deepcopy(sleep_summary)
df_sleep_summary_5_16_to_8_10_2017 = pd.DataFrame(sleep_summary_5_16_to_8_10_2017)

df_sleep_summary = pd.concat([df_sleep_summary_5_1_to_5_2_2017,
                                 df_sleep_summary_5_4_to_5_13_2017,
                                 df_sleep_summary_5_16_to_8_10_2017],
                                 ignore_index=True)

#df_sleep_summary.to_csv('df_sleep_summary_2017_02_03.csv')
#df_sleep_summary.to_csv('df_sleep_summary_2017_05_01_to_2017_08_10.csv')
# still haven't gotten this for dates after 2017_08_10



# -----------------------------------------------------------------------------
# --------------------------- get saved data ---------------------------------

# hr continuous - SKIP THIS AND GO BELOW TO CONTINUOUS BY MINUTE
#df_hr_continuous_til_2_3_17 = pd.read_csv('dt_hr_continuous_2017_02_03_copy.csv')
#del df_hr_continuous_til_2_3_17['Unnamed: 0']
#df_hr_continuous_til_2_3_17['date'] = pd.to_datetime(df_hr_continuous_til_2_3_17['date'])

# --------------------
# hr continuous by min
df_hr_continuous_min_1 = pd.read_csv('dt_hr_continuous_min_2017_02_03_copy.csv')
df_hr_continuous_min_2 = pd.read_csv('df_hr_continuous_min_2017_02_01_to_2017_05_01.csv')
df_hr_continuous_min_3 = pd.read_csv('df_hr_continuous_min_2017_05_01_to_2017_08_10.csv')

del df_hr_continuous_min_1['Unnamed: 0']
del df_hr_continuous_min_2['Unnamed: 0']
del df_hr_continuous_min_3['Unnamed: 0']

df_hr_continuous_min = pd.concat([df_hr_continuous_min_1, df_hr_continuous_min_2, df_hr_continuous_min_3], ignore_index=True)

df_hr_continuous_min['date'] = pd.to_datetime(df_hr_continuous_min['date'])
df_hr_continuous_min['date_time'] = pd.to_datetime(df_hr_continuous_min['date'].astype(str) + ' ' + df_hr_continuous_min['time'])
df_hr_continuous_min['hour'] = df_hr_continuous_min['date_time'].dt.hour
df_hr_continuous_min.head()


# --------------------
# hr resting by day
df_resting_hr_1 = pd.read_csv('df_resting_hr_2017_02_03.csv')
df_resting_hr_2 = pd.read_csv('df_resting_hr_2_1_to_4_40_2017.csv')
df_resting_hr_3 = pd.read_csv('df_resting_hr_5_1_to_8_10_2017.csv')

del df_resting_hr_1['Unnamed: 0']
del df_resting_hr_2['Unnamed: 0']
del df_resting_hr_3['Unnamed: 0']

df_resting_hr = pd.concat([df_resting_hr_1, df_resting_hr_2,
                           df_resting_hr_3], ignore_index=True)

del df_resting_hr['date_2']
df_resting_hr['date'] = pd.to_datetime(df_resting_hr['date'])

df_resting_hr['hr_resting_moving_avg'] = df_resting_hr['hr_resting'].rolling(window=20).mean()
plt.plot(df_resting_hr['date'], df_resting_hr['hr_resting_moving_avg'])
plt.xticks(rotation=50)


# --------------------
# sleep continuous -- awake or not
df_sleep_continuous_1 = pd.read_csv('df_sleep_continuous_2017_02_03.csv')
df_sleep_continuous_2 = pd.read_csv('df_sleep_continuous_2017_02_01_to_2017_05_01.csv')
df_sleep_continuous_3 = pd.read_csv('df_sleep_continuous_2017_05_01_to_2017_08_10.csv')

del df_sleep_continuous_1['Unnamed: 0']
del df_sleep_continuous_2['Unnamed: 0']
del df_sleep_continuous_3['Unnamed: 0']

df_sleep_continuous = pd.concat([df_sleep_continuous_1, df_sleep_continuous_2,
                                 df_sleep_continuous_3], ignore_index=True)

df_sleep_continuous['date'] = pd.to_datetime(df_sleep_continuous['date'])
df_sleep_continuous['date_time'] = pd.to_datetime(df_sleep_continuous['date'].astype(str) + ' ' + df_sleep_continuous['time'])
df_sleep_continuous.head()  # sleep_status: 1=asleep, 2=restless, 3=awake


# --------------------
# sleep stats by day
df_sleep_summary_1 = pd.read_csv('df_sleep_summary_2017_02_03.csv')
df_sleep_summary_2 = pd.read_csv('df_sleep_summary_2017_05_01_to_2017_08_10.csv')

del df_sleep_summary_1['Unnamed: 0']
del df_sleep_summary_2['Unnamed: 0']

df_sleep_summary = pd.concat([df_sleep_summary_1, df_sleep_summary_2], ignore_index=True)
df_sleep_summary.head()
df_sleep_summary.dtypes

df_sleep_summary['date'] = pd.to_datetime(df_sleep_summary['date'])
df_sleep_summary['date_of_sleep'] = pd.to_datetime(df_sleep_summary['date_of_sleep'])



# --------------------
# get daily questions -
df_daily_qs_early = pd.read_excel('Mood Measure (Responses).xlsx')
df_daily_qs_early.head()
df_daily_qs_early.tail()

df_daily_qs_early = df_daily_qs_early[df_daily_qs_early['What are your initials']=='AM']
df_daily_qs_early['date'] = pd.to_datetime(df_daily_qs_early['Timestamp'].dt.year.astype(str) + '-' + df_daily_qs_early['Timestamp'].dt.month.astype(str) + '-' + df_daily_qs_early['Timestamp'].dt.day.astype(str))
df_daily_qs_early['date_short'] = df_daily_qs_early['date'].dt.date
df_daily_qs_early.head()
df_daily_qs_early.tail()

# qs to use and merge with current qs?
df_daily_qs_early.dtypes
df_daily_qs_early = df_daily_qs_early[['date', 'Today, are you tired?', 'Last night, did you sleep well?']]
df_daily_qs_early['energy'] = 5 - df_daily_qs_early['Today, are you tired?']
df_daily_qs_early.rename(columns={'Last night, did you sleep well?':'sleep'}, inplace=True)
df_daily_qs_early = df_daily_qs_early[['date', 'energy', 'sleep']]


df_daily_qs_current = pd.read_csv('Daily_Measure_from_2_26_17.csv')
df_daily_qs_current.head()

df_daily_qs_current['Timestamp'] = pd.to_datetime(df_daily_qs_current['Timestamp'])
df_daily_qs_current['year'] = df_daily_qs_current['Timestamp'].dt.year.astype(str).str.split('.').str[0]
df_daily_qs_current['month'] = df_daily_qs_current['Timestamp'].dt.month.astype(str).str.split('.').str[0]
df_daily_qs_current['day'] = df_daily_qs_current['Timestamp'].dt.day.astype(str).str.split('.').str[0]
df_daily_qs_current['date'] = df_daily_qs_current['year'] + '-' + df_daily_qs_current['month'] + '-' + df_daily_qs_current['day']
df_daily_qs_current['date'].replace('nan-nan-nan', np.nan, inplace=True)
df_daily_qs_current['date'] = pd.to_datetime(df_daily_qs_current['date'])

df_daily_qs_current = df_daily_qs_current[['date', 'Right now I feel energetic.',
                                           'At some point today I felt angry.',
                                           'At some point today I had fun with someone.',
                                           'Last night, I woke up from a restorative sleep.',
                                           'Last night I had ____ alcoholic drinks',
                                           'The alcohol I drank last night was ____']]

df_daily_qs_current.columns = ['date', 'energy', 'annoyed', 'fun', 'sleep', 'alcohol', 'alcohol_type']

df_daily_qs = pd.concat([df_daily_qs_early, df_daily_qs_current], ignore_index=True)
df_daily_qs.tail(20)




# -----------------
# get activity data




# -----------------------------------------------------------------------------
# merge continuous hr and sleep data

print(len(df_hr_continuous_min))  # 401004
print(len(df_sleep_continuous))  # 155166

df_hr_continuous_min.dtypes
df_sleep_continuous.dtypes


# use df_hr_continuous_min as the base df. and map sleep info onto it
# important to resample the sleep df by min. because its minutes are
# frequently on the 30sec cycle. so doesn't merge well with the hr data

# prep df_sleep_continuous
df_sleep_continuous.index = df_sleep_continuous['date_time']
df_sleep_continuous.head(20)

df_sleep_continuous = df_sleep_continuous.resample('min').median()
# when i substitute interpolate for mean, it creates df
print(len(df_sleep_continuous))  # 451101

# sleep_status: 1=asleep, 2=restless, 3=awake
df_sleep_continuous = df_sleep_continuous[df_sleep_continuous['sleep_status'].notnull()]
print(len(df_sleep_continuous))  # 153672
df_sleep_continuous['sleep_status'].value_counts()
time_to_sleep_dict = dict(df_sleep_continuous['sleep_status'])


# prep df_hr_continuous_min
df_hr_continuous_min.index = df_hr_continuous_min['date_time']

df_hr_continuous_min['time_lag'] = df_hr_continuous_min['time'].shift(1)
df_hr_continuous_min['minute_lag'] = df_hr_continuous_min['time_lag'].str.split(':').str[1]
df_hr_continuous_min['minute'] = df_hr_continuous_min['time'].str.split(':').str[1]

df_hr_continuous_min['minute_difference'] = df_hr_continuous_min['minute'].astype(float) - df_hr_continuous_min['minute_lag'].astype(float)
df_hr_continuous_min['minute_difference'].value_counts()
df_hr_continuous_min[df_hr_continuous_min['minute_difference']==5]
df_hr_continuous_min[(df_hr_continuous_min['date_time']>'2016-10-02 02:50:00') & (df_hr_continuous_min['date_time']<'2016-10-02 03:10:00')]

df_hr_continuous_min = df_hr_continuous_min.resample('min').median()  #
df_hr_continuous_min[(df_hr_continuous_min.index>'2016-10-02 02:50:00') & (df_hr_continuous_min.index<'2016-10-02 03:10:00')]

len(df_hr_continuous_min)  # 452160
len(df_hr_continuous_min[df_hr_continuous_min['hr'].isnull()])  # these nulls are where there was no time in original df
df_hr_continuous_min['date_time'] = df_hr_continuous_min.index
df_hr_continuous_min['sleep_status'] = df_hr_continuous_min['date_time'].map(time_to_sleep_dict)
len(df_hr_continuous_min[df_hr_continuous_min['sleep_status'].isnull()])
len(df_hr_continuous_min[df_hr_continuous_min['sleep_status'].notnull()])  # 153672

# how to structure?
# want to include sleep that i get before midniight, so can't just use sleep from a particular date
# ea day at 6pm, take sleep info from past 24 hours. label that as sleep from that day

df_hr_continuous_min.head(100)
df_hr_continuous_min['date'] = df_hr_continuous_min['date_time'].dt.date
df_hr_continuous_min['date'] = pd.to_datetime(df_hr_continuous_min['date'])
df_hr_continuous_min.head()

df_groupby = df_hr_continuous_min.groupby('date').size().reset_index().rename(columns={0:'count'})
df_groupby.head()
df_groupby[df_groupby['count']!=1440]
# good - so each day has right and same number of rows


df_hr_continuous_min[1075:1085]
len(df_hr_continuous_min[df_hr_continuous_min['date_time']<'2016-10-01 18:00:00'])
# 1080 rows til 6pm of first day. 60min X 18 hours. makes sense.
min_to_lag = 1440 - 1080
min_to_lag

# create new date variable -- it's lagged 360 min, i.e., 6 hours, i.e., starting 6pm instead of midnight
df_hr_continuous_min['date_sleep'] = df_hr_continuous_min['date'].shift(-360)
df_hr_continuous_min[1070:1100]
df_hr_continuous_min[50030:50050]
# now if group by this lagged date, it'll give all the sleep from night before
# as long as started sleeping after 6pm the night before


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




# -----------------------------------------------------------------------------
sns.lmplot(x='sleep_subjective_rev', y='hr_mean_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})
sns.lmplot(x='sleep_subjective_rev', y='hr_std_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})
sns.lmplot(x='sleep_subjective_rev', y='outlier_mean_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})
sns.lmplot(x='sleep_subjective_rev', y='hr_deviation_from_smoothed_mean_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})
sns.lmplot(x='sleep_subjective_rev', y='hr_deviation_from_smoothed_std_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})
sns.lmplot(x='sleep_subjective_rev', y='hr_rolling_min_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})
sns.lmplot(x='sleep_subjective_rev', y='hr_rolling_max_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})
sns.lmplot(x='sleep_subjective_rev', y='hr_slope_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})
sns.lmplot(x='sleep_subjective_rev', y='minutes_asleep_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})
sns.lmplot(x='sleep_subjective_rev', y='minutes_awake_z', data=df_nightly_metrics_joined[df_nightly_metrics_joined['sleep_subjective_rev'].notnull()], scatter_kws={'alpha':.2})





















# -----------
# play around

# framework. could use for prsentation too:
# look at several different time series plot of hr over course of a night (like on the fitbit dashboard)
# see how diff and come up w several hypotheses about how can look at hr to predict how satisfied i feel with sleep in morning.
# hr in the last hour or so. hr ups and downs (could signal restless sleep)
# hr stdev. etc.


df_hr_continuous_min.head()
df_hr_continuous_min['date_time']  # merge w continuous sleep on date_time?
df_hr_continuous_min['ibi'] = (60/df_hr_continuous_min['hr']*1000)
df_hr_continuous_min.head()
len(df_hr_continuous_min)  # 151259

df_sleep_continuous.dtypes
#df_sleep_continuous['date_time'] = df_sleep_continuous['date_time'].values.astype('<M8[m]')
df_sleep_continuous.head()
df_hr_continuous_min.head()
len(df_sleep_continuous)  # 60389

# turn indices into date
df_hr_continuous_min.index = df_hr_continuous_min['date_time']
df_hr_by_min = df_hr_continuous_min.resample('min').interpolate()  # when i add interpolate, it creates df
len(df_hr_by_min)  # 181440

# was important to resample the sleep df by min too. because it's minutes were
# frequently on the 30sec cycle. so wasn't merging well with the hr data!
df_sleep_continuous.index = df_sleep_continuous['date_time']
df_sleep_by_min = df_sleep_continuous.resample('min').mean()  # when i add interpolate, it creates df
df_sleep_by_min['sleep_status'].value_counts()
len(df_sleep_by_min[df_sleep_by_min['sleep_status'].isnull()])  # 120004
len(df_sleep_by_min)  # 180393

time_to_sleep_dict = dict(df_sleep_by_min['sleep_status'])
df_hr_by_min['date_time'] = df_hr_by_min.index
df_hr_by_min['sleep_status'] = df_hr_by_min['date_time'].map(time_to_sleep_dict)
df_hr_by_min['sleep_status'].value_counts()

df_hr_by_min_asleep = df_hr_by_min[df_hr_by_min['sleep_status']==1]
df_hr_by_min_asleep_by_day = df_hr_by_min_asleep.resample('D').mean().interpolate()

plt.plot(df_hr_by_min_asleep_by_day['hr'])
plt.xticks(rotation=45)
plt.ylim(49,63)

df_hr_by_min_asleep_by_week = df_hr_by_min_asleep.resample('W').mean().interpolate()
plt.plot(df_hr_by_min_asleep_by_week['hr'])
plt.xticks(rotation=45)
plt.ylim(49,63)

df_hr_by_min_asleep_by_month = df_hr_by_min_asleep.resample('M').mean().interpolate()
plt.plot(df_hr_by_min_asleep_by_month['hr'])
plt.xticks(rotation=30)
plt.ylim(49,63)

df_hr_by_min_asleep_by_week_std = df_hr_by_min_asleep.resample('W').std().interpolate()
plt.plot(df_hr_by_min_asleep_by_week_std['hr'])
plt.xticks(rotation=30)



# plot hr for one night (like in fitbit dash). so can compare a bunch of
# nights to come up with several hypoths to test. i.e., plot a bunch of
# nights, see what varies between them, and whatever that is becomes a var
# to examine in relation to other vars, e.g., daily ratings.

df_hr_by_min_asleep.head()
df_hr_by_min_asleep['date_short'] = pd.to_datetime(df_hr_by_min_asleep.loc[:, 'date_time'].dt.year.astype(str) + '-' + df_hr_by_min_asleep.loc[:, 'date_time'].dt.month.astype(str) + '-' + df_hr_by_min_asleep.loc[:, 'date_time'].dt.day.astype(str))
#df_hr_by_min_asleep['date_short'] = df_hr_by_min_asleep.loc[:, 'date_time'].dt.date
df_hr_by_min_asleep['time'] = df_hr_by_min_asleep.loc[:, 'date_time'].dt.time

#df_hr_by_min_asleep['year'] = df_hr_by_min_asleep.loc[:, 'date_time'].dt.year
#df_hr_by_min_asleep['month'] = df_hr_by_min_asleep.loc[:, 'date_time'].dt.month
#df_hr_by_min_asleep['day'] = df_hr_by_min_asleep.loc[:, 'date_time'].dt.day

df_hr_by_min_asleep_day = df_hr_by_min_asleep[df_hr_by_min_asleep['date_short']=='2016-10-01']
df_hr_by_min_asleep_day['hr_rolling'] = df_hr_by_min_asleep_day['hr'].rolling(window=30).mean()
plt.plot(df_hr_by_min_asleep_day['hr_rolling'])
plt.ylabel('Heart Rate', fontsize=18)
plt.xlabel('Time', fontsize=18)
sns.despine()

df_hr_by_min_asleep_day_5min = df_hr_by_min_asleep_day.resample('5min').mean().interpolate()
plt.plot(df_hr_by_min_asleep_day_5min['hr'])


# overlay several nights
dates = pd.date_range('2016-10-01', '2016-12-30', freq='D')
for day in dates[:]:
    df_hr_by_min_asleep_day = df_hr_by_min_asleep[df_hr_by_min_asleep['date_short']==day]
    df_hr_by_min_asleep_day['hr_rolling'] = df_hr_by_min_asleep_day['hr'].rolling(window=15).mean()
    #df_hr_by_min_asleep_day = df_hr_by_min_asleep_day.resample('5min').mean()
    #df_hr_by_min_asleep_day['date_time'] = df_hr_by_min_asleep_day.index
    #df_hr_by_min_asleep_day.loc[:, 'time'] = df_hr_by_min_asleep_day.loc[:, 'date_time'].dt.time
    plt.plot(df_hr_by_min_asleep_day['time'], df_hr_by_min_asleep_day['hr_rolling'], alpha=.1)
    plt.ylabel('Hear rTate', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.xlim('00:00:00', '09:00:00')
    plt.ylim(40,90)
    plt.title('heart rate over the course of a night', fontsize=20)
    sns.despine()
plt.show()


# ideas for these vars:  hr just before waking up
# or hr fluctuation between hours that asleep and restless?
# actually, that looks interesting. corr std with waking ratings of mood, etc.
# or amount restless and away between sleep?
# and incorporate activity

# select hr starting at 4am
df_hr_by_min_asleep['cutoff_time'] = pd.to_datetime()
df_hr_by_min_asleep_after_4 = df_hr_by_min_asleep[(df_hr_by_min_asleep['date_time'].dt.hour > 4) & (df_hr_by_min_asleep['date_time'].dt.hour < 10)]

df_hr_by_min_asleep_after_4_by_week = df_hr_by_min_asleep_after_4.resample('D').mean().interpolate()
plt.plot(df_hr_by_min_asleep_after_4_by_week['hr'])
plt.xticks(rotation=45)
plt.ylim(47,65)

df_hr_by_min_asleep_after_4_by_week = df_hr_by_min_asleep_after_4.resample('W').std()
plt.plot(df_hr_by_min_asleep_after_4_by_week['hr'])
plt.xticks(rotation=45)
plt.ylim(2,12)


# map on daily ratings
df_daily_qs.head()
date_to_mood_dict = dict(zip(df_daily_qs['date'], df_daily_qs['mood_positive']))
date_to_subjective_sleep_dict = dict(zip(df_daily_qs['date'], df_daily_qs['sleep_well']))

df_hr_by_min_asleep.loc[:, 'mood_positive'] = df_hr_by_min_asleep.loc[:, 'date_short'].map(date_to_mood_dict)
df_hr_by_min_asleep.loc[:, 'sleep_well'] = df_hr_by_min_asleep.loc[:, 'date_short'].map(date_to_subjective_sleep_dict)

plt.hist(df_daily_qs['mood_positive'])
df_daily_qs['mood_positive'].value_counts()
plt.hist(df_daily_qs['sleep_well'])
df_daily_qs['sleep_well'].value_counts()


# this looks cool -- work with this plot below
# ((plot ea night but color diff depending on a daily rating))
dates = pd.date_range('2016-10-01', '2017-2-28', freq='D')
dates = df_daily_qs['date'].dt.date
for day in dates[19:20]:
    print(day)
    alpha=.1
    daily_rating = df_daily_qs[df_daily_qs['date']==day]
    if daily_rating['sleep_well'].values > 4:
        color = 'green'
        alpha = (daily_rating['sleep_well'].values / 7) * .325
    elif daily_rating['sleep_well'].values <= 4:
        color = 'red'
        alpha = ((8 - daily_rating['sleep_well'].values) / 7) * .325
    print(color)

    df_hr_by_min_asleep_day = df_hr_by_min_asleep[df_hr_by_min_asleep['date_short']==day]

#    # do this on entire df first -- to try and get rid of big jumps that seem like it's from me waking up
#    df_hr_by_min_asleep_day.loc[:, 'hr_prior'] = df_hr_by_min_asleep_day.loc[:, 'hr'].shift(1)
#    df_hr_by_min_asleep_day.loc[:, 'hr_prior_2'] = df_hr_by_min_asleep_day.loc[:, 'hr'].shift(2)
#    df_hr_by_min_asleep_day['big_change'] = 0
#    df_hr_by_min_asleep_day.loc[df_hr_by_min_asleep_day['hr'] > (df_hr_by_min_asleep_day['hr_prior']+10), 'big_change'] = 1
#    df_hr_by_min_asleep_day.loc[df_hr_by_min_asleep_day['hr'] > (df_hr_by_min_asleep_day['hr_prior_2']+10), 'big_change'] = 1
#    df_hr_by_min_asleep_day.loc[df_hr_by_min_asleep_day['big_change']==1, 'hr'] = df_hr_by_min_asleep_day[['hr', 'hr_prior', 'hr_prior_2']].mean(axis=1)

    df_hr_by_min_asleep_day['hr_rolling'] = df_hr_by_min_asleep_day['hr'].rolling(window=30, center=True).mean()
    df_hr_by_min_asleep_day = df_hr_by_min_asleep_day.resample('2min').mean()
    df_hr_by_min_asleep_day['date_time'] = df_hr_by_min_asleep_day.index
    df_hr_by_min_asleep_day.loc[:, 'time'] = df_hr_by_min_asleep_day.loc[:, 'date_time'].dt.time
    plt.plot(df_hr_by_min_asleep_day['time'], df_hr_by_min_asleep_day['hr'], alpha=alpha, color=color)
    plt.plot(df_hr_by_min_asleep_day['time'], df_hr_by_min_asleep_day['hr_rolling'], alpha=alpha, color='black')
    plt.ylabel('Heart Rate', fontsize=18)
    plt.xlabel('Time', fontsize=18)
    plt.xlim('00:00:00', '09:00:00')
    plt.ylim(40,90)
    plt.title('Heart Rate Over the Course of a Night', fontsize=20)
    sns.despine()
plt.show()


# could decide that if the raw hr by min is more than x beats higher than a 30min
# rolling avg, that it's me waking up. more or less than 10 beats?
# pre-process all of this to clean it. then play with plotting it in diff ways
# cool.



#df_sleep_continuous.index = df_sleep_continuous['date_time']


## resample each df
#df_sleep_by_min = df_sleep_continuous.resample('min').interpolate()  # when i add interpolate, it creates df
#df_sleep_by_min['date_time'] = df_sleep_by_min.index
#df_hr_by_min = df_hr_continuous_min.resample('min').interpolate()  # when i add interpolate, it creates df
#df_hr_by_min['date_time'] = df_hr_by_min.index
#
#df_sleep_hr_merged_by_min = pd.merge(df_sleep_by_min, df_hr_by_min, on='date_time', how='outer')


df_hr_sleep_continuous = pd.merge(df_hr_continuous_min, df_sleep_continuous, on='date_time', how='outer')
len(df_hr_sleep_continuous)  # 181899

df_hr_sleep_continuous.head(5)
df_hr_sleep_continuous.columns
df_hr_sleep_continuous = df_hr_sleep_continuous[['date_x', 'hr', 'time_x', 'date_time',
'hour', 'ibi', 'sleep_status', ]]
df_hr_sleep_continuous.head()

df_hr_sleep_continuous[df_hr_sleep_continuous['ibi'].isnull()]
# 151279    NaT NaN    NaN 2016-10-02 00:08:00   NaN  NaN           1.0

# find something near this data point above to see what's going on
df_hr_day_1 = df_hr_continuous_min[df_hr_continuous_min['date']=='2016-10-02']
df_hr_day_1.head(40)
# the time is just missing. i.e., didn't get a hr reading. so should average the
# two readings from either side? makes some sense. fill in these gaps.
# to do, create a new df with every minute.

# reindex vs resample. probably resample is what i want.
df_hr_sleep_continuous.index = df_hr_sleep_continuous['date_time']
df_hr_sleep_continuous.head(10)

df_by_min = df_hr_sleep_continuous.resample('min').interpolate()  # when i add interpolate, it creates df
#df_by_min = df_hr_sleep_continuous.resample('min').mean()  # when i add interpolate, it creates df
df_by_min['date'] = df_by_min.index.normalize()
df_hr_day_2 = df_by_min[df_by_min['date']=='2016-10-02']
df_hr_day_2.head()

df_by_min.shape, df_hr_sleep_continuous.shape
# doesn't make sense. suggests that maybe
#df_hr_day_2.shape, df_hr_day_1.shape


df_by_min['sleep_status'].value_counts()

df_by_min.index.min()
df_hr_sleep_continuous.index.min()


low_hr_by_day_dict = {'date':[], 'hr_lowest_hour':[]}
for day in df_hr_continuous_min['date'].unique():
    print(day)
    df_day = df_hr_continuous_min[df_hr_continuous_min['date']==day]
    df_day_sorted = df_day.sort_values(by='hr')
    df_day_sorted = df_day_sorted.reset_index(drop=True)
    df_day_sorted_lowest_60 = df_day_sorted[df_day_sorted.index<60]
    mean_hr_lowest_hour = df_day_sorted_lowest_60['hr'].mean()
    low_hr_by_day_dict['date'].append(day)
    low_hr_by_day_dict['hr_lowest_hour'].append(mean_hr_lowest_hour)

df_hr_low_by_day = pd.DataFrame(low_hr_by_day_dict)
df_hr_low_by_day.head()
plt.plot(df_hr_low_by_day['date'], df_hr_low_by_day['hr_lowest_hour'])
df_hr_low_by_day['hr_low_moving_avg'] = df_hr_low_by_day['hr_lowest_hour'].rolling(window=5).mean()
df_hr_low_by_day['hr_low_moving_avg_10'] = df_hr_low_by_day['hr_lowest_hour'].rolling(window=10).mean()

plt.plot(df_hr_low_by_day['date'], df_hr_low_by_day['hr_low_moving_avg'], alpha=.7)
plt.xticks(rotation=50)

plt.plot(df_hr_low_by_day['date'], df_hr_low_by_day['hr_low_moving_avg_10'], alpha=.7)
plt.xticks(rotation=50)

# qs: does this corr with exercise? mood? se? subj sleep qual?
# sleep hours in day(s) before? weighted avg of sleep qual or hours might be
# best way to compute?

df_hr_low_by_day.head()

df_merge = pd.merge(df_resting_hr, df_daily_qs, on=['date'], how='left')
df_merge = pd.merge(df_hr_low_by_day, df_daily_qs, on=['date'], how='left')
df_merge.head()
df_merge['hr_lowest_day_before'] = df_merge['hr_lowest_hour'].shift(1)
df_merge['hr_lowest_day_after'] = df_merge['hr_lowest_hour'].shift(-1)

# this one makes some sense
sns.lmplot(x='hr_lowest_hour', y='Today, are you tired?', data=df_merge, lowess=True, y_jitter=.25, scatter_kws={'alpha':.25, 's':50})  # , color='red'
plt.xlim(44,54)
plt.ylim(.5,6.5)

sns.lmplot(x='hr_lowest_hour', y='Today, is your self-esteem high?', data=df_merge, lowess=True, y_jitter=.25, scatter_kws={'alpha':.25, 's':50})  # , color='r'
plt.xlim(44,54)
plt.ylim(3.5,7)

sns.lmplot(x='hr_lowest_hour', y='Today, is your mood positive?', data=df_merge, lowess=True, y_jitter=.25, scatter_kws={'alpha':.25, 's':50}, color='r')
plt.xlim(44,54)
plt.ylim(3.5,7)

sns.lmplot(x='hr_lowest_hour', y='Last night, did you sleep well?', data=df_merge, lowess=True, y_jitter=.25, scatter_kws={'alpha':.25, 's':50}, color='r')
plt.xlim(44,54)
plt.ylim(.5,7.2)

sns.lmplot(x='hr_lowest_day_before', y='hr_lowest_day_after', data=df_merge, lowess=True, y_jitter=.25, scatter_kws={'alpha':.25, 's':50})  # , color='r'
plt.xlim(44,54)
plt.ylim(44,54)

fig, axes = plt.subplots(ncols=1)
sns.regplot(x='hr_lowest_day_before', y='hr_lowest_hour', data=df_merge, y_jitter=.25, scatter_kws={'alpha':.25, 's':50}, ax=axes, lowess=True)
sns.regplot(x='hr_lowest_day_before', y='hr_lowest_day_after', data=df_merge, y_jitter=.25, scatter_kws={'alpha':.25, 's':50}, ax=axes, lowess=True)


plt.plot(df_hr_low_by_day['date'], df_hr_low_by_day['hr_low_moving_avg_10'], alpha=.7)
plt.xticks(rotation=50)


# examine daily qs
df_daily_qs.head()
df_daily_qs['mood'] = df_daily_qs['Today, is your mood positive?']
df_daily_qs['se'] = df_daily_qs['Today, is your self-esteem high?']
df_daily_qs['tired'] = df_daily_qs['Today, are you tired?']
df_daily_qs['sleep'] = df_daily_qs['Last night, did you sleep well?']
df_daily_qs['initials'] = df_daily_qs['What are your initials']
df_daily_qs[['mood', 'se', 'tired', 'sleep']].corr()


df_mike = df_daily_qs[df_daily_qs['initials']=='MJ']
df_andy = df_daily_qs[df_daily_qs['initials']=='AM']

df_andy['se_moving_avg'] = df_andy['se'].rolling(window=15).mean()
plt.plot(df_andy['date'], df_andy['se_moving_avg'], label='andy')
plt.xticks(rotation=50)
plt.ylim(3,6)

df_mike['se_moving_avg'] = df_mike['se'].rolling(window=15).mean()
plt.plot(df_mike['date'], df_mike['se_moving_avg'], label='mike')
plt.xticks(rotation=50)
plt.ylim(3,6)


df_mike['tired_moving_avg'] = df_mike['tired'].rolling(window=15).mean()
df_andy['tired_moving_avg'] = df_andy['tired'].rolling(window=15).mean()
plt.plot(df_andy['date'], df_andy['tired_moving_avg'], label='andy')
plt.xticks(rotation=50)
plt.ylim(2,4)
plt.plot(df_mike['date'], df_mike['tired_moving_avg'], label='mike')
plt.xticks(rotation=50)
plt.ylim(2,4)
plt.legend(fontsize=15)

df_mike['sleep_moving_avg'] = df_mike['sleep'].rolling(window=20).mean()
df_andy['sleep_moving_avg'] = df_andy['sleep'].rolling(window=20).mean()
plt.plot(df_andy['date'], df_andy['sleep_moving_avg'], label='andy', linewidth=5, alpha=.5)
plt.xticks(rotation=50)
plt.ylim(3.5,5)
plt.plot(df_mike['date'], df_mike['sleep_moving_avg'], label='mike', linewidth=5, alpha=.5)
plt.xticks(rotation=50)
plt.ylim(3.5,5)
plt.legend(fontsize=15)



plt.plot(df_mike['date'], df_mike['tired_moving_avg'])
plt.xticks(rotation=50)


sns.barplot(x='initials', y='se', data=df_daily_qs, color='orange')


plt.scatter(df_daily_qs['Today, is your mood positive?'], df_daily_qs['Today, is your self-esteem high?'], alpha=.3)
sns.lmplot(x='Today, is your mood positive?', y='Today, is your self-esteem high?', data=df_daily_qs)
df_daily_qs[['Today, is your mood positive?', 'Today, is your self-esteem high?', 'Today, are you tired?']].corr()
sns.lmplot(x='Today, is your mood positive?', y='Today, are you tired?', data=df_daily_qs)

results = smf.ols(formula = 'tired ~ se + mood', data=df_daily_qs).fit()
print(results.summary())






df_daily_qs['se_moving_avg_10'] = df_daily_qs['Today, is your self-esteem high?'].rolling(window=10).mean()
df_daily_qs['mood_moving_avg_10'] = df_daily_qs['Today, is your mood positive?'].rolling(window=10).mean()
df_daily_qs['sleep_moving_avg_10'] = df_daily_qs['Last night, did you sleep well?'].rolling(window=10).mean()
df_daily_qs['tired_moving_avg_10'] = df_daily_qs['Today, are you tired?'].rolling(window=10).mean()

df_daily_qs['sleep_poor'] = np.nan
df_daily_qs.loc[df_daily_qs['Last night, did you sleep well?'] < 4 , 'sleep_poor'] = 1
df_daily_qs.loc[df_daily_qs['Last night, did you sleep well?'] >= 4 , 'sleep_poor'] = 0
df_daily_qs['sleep_poor_moving_avg_10'] = df_daily_qs['sleep_poor'].rolling(window=10).mean()

df_daily_qs['sleep_well'] = np.nan
df_daily_qs.loc[df_daily_qs['Last night, did you sleep well?'] > 4 , 'sleep_well'] = 1
df_daily_qs.loc[df_daily_qs['Last night, did you sleep well?'] <= 4 , 'sleep_well'] = 0
df_daily_qs['sleep_well_moving_avg_10'] = df_daily_qs['sleep_well'].rolling(window=10).mean()


plt.plot(df_daily_qs['date'], df_daily_qs['se_moving_avg_10'], alpha=.7)
plt.xticks(rotation=50)

plt.plot(df_daily_qs['date'], df_daily_qs['mood_moving_avg_10'], alpha=.7)
plt.xticks(rotation=50)

plt.plot(df_daily_qs['date'], df_daily_qs['sleep_moving_avg_10'], alpha=.7)
plt.xticks(rotation=50)

plt.plot(df_daily_qs['date'], df_daily_qs['sleep_poor_moving_avg_10'], alpha=.7)
plt.xticks(rotation=50)

plt.plot(df_daily_qs['date'], df_daily_qs['sleep_well_moving_avg_10'], alpha=.7)
plt.xticks(rotation=50)

plt.plot(df_daily_qs['date'], df_daily_qs['tired_moving_avg_10'], alpha=.7)
plt.xticks(rotation=50)


df_hr_low_by_day['hr_low_moving_avg_10_v2'] = df_hr_low_by_day['hr_low_moving_avg_10'] * .05 - 2
plt.plot(df_hr_low_by_day['date'], df_hr_low_by_day['hr_low_moving_avg_10_v2'], alpha=.7)
plt.xticks(rotation=50)
plt.plot(df_daily_qs['date'], df_daily_qs['sleep_poor_moving_avg_10'], alpha=.7)
plt.xticks(rotation=50)










# ----
# anys

df_merge = pd.merge(df_resting_hr, df_daily_qs, on=['date'], how='left')
df_merge.head()

plt.scatter(df_merge['hr_resting'], df_merge['Today, is your mood positive?'], alpha=.2)
plt.scatter(df_merge['hr_resting'], df_merge['Today, is your self-esteem high?'], alpha=.2)
plt.scatter(df_merge['hr_resting'], df_merge['Today, are you tired?'], alpha=.2)
plt.scatter(df_merge['hr_resting'], df_merge['Last night, did you sleep well?'], alpha=.2)

sns.lmplot(x='hr_resting', y='Today, is your mood positive?', data=df_merge, order=1)
sns.lmplot(x='hr_resting', y='Today, is your mood positive?', data=df_merge, order=2)
sns.lmplot(x='hr_resting', y='Today, is your mood positive?', data=df_merge, order=3)

sns.set_style('whitegrid')
sns.barplot(x='hr_resting', y='Today, is your mood positive?', data=df_merge)
#plt.xlim(52,59)
plt.ylim(1,7.5)
sns.despine()

sns.barplot(x='hr_resting', y='Today, is your self-esteem high?', data=df_merge)
#plt.xlim(52,59)
plt.ylim(1,7.5)
sns.despine()

sns.barplot(x='hr_resting', y='Today, are you tired?', data=df_merge)
#plt.xlim(52,59)
plt.ylim(1,7.5)
sns.despine()

sns.barplot(x='hr_resting', y='Last night, did you sleep well?', data=df_merge)
#plt.xlim(52,59)
plt.ylim(1,7.5)
sns.despine()

sns.barplot(x='Last night, did you sleep well?', y='hr_resting', data=df_merge)
#plt.xlim(52,59)
plt.ylim(50,59)
sns.despine()





# --------------------------------------- tests ---------------------------------------

sleep_ts = authd_client.time_series('sleep/minutesAsleep', period='3m')
sleep_ts['sleep-minutesAsleep'][-7]

dates = pd.date_range('10/01/2016', '02/03/17', freq='D')

date = dates[53]
activity = authd_client.time_series('activity', period='1d')

sleep_summary_day = {'awake_count':[], 'awake_duration':[], 'awakenings_count':[],
                     'date_of_sleep':[], 'duration':[], 'efficiency':[], 'is_main_sleep':[],
                    'minutes_asleep':[], 'minutes_awake':[], 'minutes_to_fall_asleep':[],
                    'restless_count':[], 'restless_duration':[], 'total_minutes_asleep':[],
                    'total_sleep_records':[], 'total_time_in_bed':[]}
for date in dates[51:]:
date = dates[112]
    date = str(date.year) + '-' + date.strftime('%m') + '-' + date.strftime('%d')
    print(date)
    stats_activity = authd_client.activities(date=date)


dir(authd_client)
help(authd_client)
help(authd_client.time_series)
help(authd_client.intraday_time_series)
help(authd_client.sleep)




date = '2018-06-01'
stats_hr = authd_client.intraday_time_series('activities/heart', base_date='2016-12-01', detail_level='1sec', start_time='12:00', end_time='12:10')
stats_hr = authd_client.intraday_time_series('activities/sleep', base_date='2016-12-01', detail_level='1sec', start_time='12:00', end_time='12:10')

help(stats_hr)
dir(stats_hr)

dir(authd_client.sleep(pd.to_datetime('2018-06-01')))
help(authd_client.get_sleep(pd.to_datetime('2018-06-01')))

x = authd_client.get_sleep(date=pd.to_datetime('2018-06-02'))
x.values()


x = authd_client.get_sleep(date=pd.to_datetime('2018-06-01'))
x.items()
help(x)

sleep = authd_client.sleep('', base_date='2016-12-01', detail_level='1sec', start_time='12:00', end_time='12:10')
sleep = authd_client.sleep('', base_date='2016-12-01', detail_level='1sec', start_time='12:00', end_time='12:10')



# continuous hr
stats_hr = authd_client.intraday_time_series('activities/heart', base_date='2016-12-01', detail_level='1sec', start_time='12:00', end_time='12:10')
stats_hr = authd_client.intraday_time_series('activities/heart', base_date='2016-09-28', detail_level='1sec', start_time='00:00', end_time='23:59')



hr_data = stats_hr['activities-heart-intraday']['dataset']


# resting hr
stats_resting_hr = authd_client.time_series('activities/heart', base_date='2016-12-01', period='1d')
stats_resting_hr['activities-heart'][0]['dateTime']
stats_resting_hr['activities-heart'][0]['value']['restingHeartRate']


# sleep
# get sleep so i can see when i fell asleep -- so get hr during the night, during sleep.
# note: values for minuteData can be 1 ("asleep"), 2 ("awake"), or 3 ("really awake").
stats_sleep = authd_client.sleep(date='2018-4-30')
stats_sleep = authd_client.get_sleep(date=pd.to_datetime('2018-03-02'))

# now these give whether woke and for how long
# and give start of sleep stagea and how long


stats_sleep = authd_client.sleep(date='2017-01-31')


stats_sleep['sleep'][0]['awakeCount']
stats_sleep['sleep'][0]['awakeDuration']
stats_sleep['sleep'][0]['awakeningsCount']
stats_sleep['sleep'][0]['dateOfSleep']
stats_sleep['sleep'][0]['duration']
stats_sleep['sleep'][0]['efficiency']
stats_sleep['sleep'][0]['isMainSleep']
stats_sleep['sleep'][0]['minutesAsleep']
stats_sleep['sleep'][0]['minutesAwake']
stats_sleep['sleep'][0]['minutesToFallAsleep']
stats_sleep['sleep'][0]['restlessCount']
stats_sleep['sleep'][0]['restlessDuration']
stats_sleep['summary']
stats_sleep['sleep'][0]['minuteData']  # min by min whether asleep or not
