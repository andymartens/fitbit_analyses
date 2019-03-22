#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:00:41 2019

@author: charlesmartens


Analyses
* with each row daily numbers. start w this for now.
* but would be to plot timeseries divived by IVs too, and controlling for
  confounds, e.g., the 3 alcohol timeseries but regress the covariates out 
  of the minute-by-minute hr metric. nice way to viz the causal patterns.


"""



cd /Users/charlesmartens/Documents/projects/fitbit_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import statsmodels.formula.api as smf #with this 'formula' api, don't need to create the design #matrices (does it automatically).
from statsmodels.formula.api import *
from matplotlib.dates import DateFormatter
import pickle
from datetime import datetime, timedelta
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
#import csv
i#mport sys, os
#from collections import deque
#import configparser

sns.set_style('white')



# -----
# import analytic file foundation
df_hr_mean = pd.read_pickle('df_by_day_for_anys.pkl')
df_hr_mean.shape  # (479, 2)

# should sort
df_hr_mean = df_hr_mean.sort_values(by='date_sleep')
df_hr_mean.head()

# rename sleeping hr
df_hr_mean.rename(columns={'hr_mean':'sleeping_hr'}, inplace=True)


# resample so each day has a row? I think that's prob important?
len(df_hr_mean.resample('1D', on='date_sleep').reset_index())  # 509
df_hr_mean = df_hr_mean.resample('1D', on='date_sleep').mean().reset_index()

len(df_hr_mean[df_hr_mean['hr_mean'].isnull()]) / len(df_hr_mean)

def create_lagged_past_date_variables(number_of_lags, date_variable, df):
    for lag in list(range(1,number_of_lags+1)):
        df['date_past_'+str(lag)] = df[date_variable] - timedelta(days=lag)
    return df

def create_lagged_future_date_variables(number_of_lags, date_variable, df):
    for lag in list(range(1,number_of_lags+1)):
        df['date_forward_'+str(lag)] = df[date_variable] + timedelta(days=lag)
    return df

df_hr_mean = create_lagged_future_date_variables(1, 'date_sleep', df_hr_mean)
df_hr_mean = create_lagged_past_date_variables(1, 'date_sleep', df_hr_mean)
df_hr_mean = create_lagged_past_date_variables(2, 'date_sleep', df_hr_mean)
df_hr_mean = create_lagged_past_date_variables(3, 'date_sleep', df_hr_mean)
df_hr_mean = create_lagged_past_date_variables(4, 'date_sleep', df_hr_mean)
df_hr_mean = create_lagged_past_date_variables(5, 'date_sleep', df_hr_mean)


date_sleep_to_sleeping_hr_dict = dict(zip(df_hr_mean['date_sleep'], df_hr_mean['sleeping_hr']))
df_hr_mean['sleeping_hr_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_sleeping_hr_dict)
df_hr_mean['sleeping_hr_lag2'] = df_hr_mean['date_past_2'].map(date_sleep_to_sleeping_hr_dict)
df_hr_mean['sleeping_hr_lag3'] = df_hr_mean['date_past_3'].map(date_sleep_to_sleeping_hr_dict)
df_hr_mean['sleeping_hr_lag4'] = df_hr_mean['date_past_4'].map(date_sleep_to_sleeping_hr_dict)


# -----
# import dicts with info to map onto analytic file

# naming convention: 
# with the idea that resting hr will be dv
# and sleep metrics will be the main ivs.
# sleep metrics mapped to date_sleep = no suffix
# sleep metrics mapped to prior dates = lag1, lag2, etc.
# metrics when awake (including resting hr and alcohol) mapped to the day after date_sleep = no suffix
# metrics when awake mapped to date_sleep = lag1
# metrics when awake mapped to prior dates = lag2, lag3, etc.

df_hr_mean.columns

# hr intensity metrics
with open('date_to_hr_highest_levels_30_min_dict.pkl', 'rb') as picklefile:
    date_to_hr_highest_levels_30_min_dict = pickle.load(picklefile)
with open('date_to_hr_highest_levels_60_min_dict.pkl', 'rb') as picklefile:
    date_to_hr_highest_levels_60_min_dict = pickle.load(picklefile)

df_hr_mean['hr_high_30_lag1'] = df_hr_mean['date_sleep'].map(date_to_hr_highest_levels_30_min_dict)
df_hr_mean['hr_high_60_lag1'] = df_hr_mean['date_sleep'].map(date_to_hr_highest_levels_60_min_dict)
df_hr_mean['hr_high_30_lag2'] = df_hr_mean['date_past_1'].map(date_to_hr_highest_levels_30_min_dict)
df_hr_mean['hr_high_60_lag2'] = df_hr_mean['date_past_1'].map(date_to_hr_highest_levels_60_min_dict)


#len(df_hr_mean[df_hr_mean['hr_high_30_minutes'].isnull()]) # 2
#len(df_hr_mean[df_hr_mean['hr_high_60_minutes'].isnull()]) # 2
#df_hr_mean[['hr_high_30_minutes', 'hr_high_60_minutes']].corr()

# hr before sleep
with open('date_sleep_to_hr_30_min_before_sleep_lag_10_dict.pkl', 'rb') as picklefile:
    date_sleep_to_hr_30_min_before_sleep_lag_10_dict = pickle.load(picklefile)
with open('date_sleep_to_hr_60_min_before_sleep_lag_10_dict.pkl', 'rb') as picklefile:
    date_sleep_to_hr_60_min_before_sleep_lag_10_dict = pickle.load(picklefile)
with open('date_sleep_to_hr_120_min_before_sleep_lag_10_dict.pkl', 'rb') as picklefile:
    date_sleep_to_hr_120_min_before_sleep_lag_10_dict = pickle.load(picklefile)

df_hr_mean['hr_before_sleep_30_lag1'] = df_hr_mean['date_sleep'].map(date_sleep_to_hr_30_min_before_sleep_lag_10_dict)
df_hr_mean['hr_before_sleep_60_lag1'] = df_hr_mean['date_sleep'].map(date_sleep_to_hr_60_min_before_sleep_lag_10_dict)
df_hr_mean['hr_before_sleep_120_lag1'] = df_hr_mean['date_sleep'].map(date_sleep_to_hr_120_min_before_sleep_lag_10_dict)

df_hr_mean['hr_before_sleep_30_lag2'] = df_hr_mean['date_past_1'].map(date_sleep_to_hr_30_min_before_sleep_lag_10_dict)
df_hr_mean['hr_before_sleep_60_lag2'] = df_hr_mean['date_past_1'].map(date_sleep_to_hr_60_min_before_sleep_lag_10_dict)
df_hr_mean['hr_before_sleep_120_lag2'] = df_hr_mean['date_past_1'].map(date_sleep_to_hr_120_min_before_sleep_lag_10_dict)

len(df_hr_mean[df_hr_mean['hr_before_sleep_60_lag1'].isnull()])  # 27

# resting hr
# df with sedentary hr mean ea day as measure of resting hr
# computed by getting the mean sedentary hr each hour and then taking
# the average of these hourly measurements
df_sedentary_hr_mean_day = pd.read_pickle('df_sedentary_hr_mean_day.pkl')
# df with minumum sedentary hr mean ea day as measure of resting hr
# computed by getting the miniumy sedentary hr each hour and then taking
# the average of these hourly measurements
df_minimum_hr_mean_day = pd.read_pickle('df_minimum_hr_mean_day.pkl')

date_to_hr_resting_dict = dict(df_sedentary_hr_mean_day)
date_to_hr_resting_minimum_dict = dict(df_minimum_hr_mean_day)

df_hr_mean['resting_hr'] = df_hr_mean['date_forward_1'].map(date_to_hr_resting_dict)
df_hr_mean['resting_min_hr'] = df_hr_mean['date_forward_1'].map(date_to_hr_resting_minimum_dict)
len(df_hr_mean[df_hr_mean['resting_hr'].isnull()])  # 8
df_hr_mean[['resting_hr', 'resting_min_hr']].corr()

df_hr_mean['resting_hr_lag1'] = df_hr_mean['date_sleep'].map(date_to_hr_resting_dict)
df_hr_mean['resting_min_hr_lag1'] = df_hr_mean['date_sleep'].map(date_to_hr_resting_minimum_dict)
df_hr_mean['resting_hr_lag2'] = df_hr_mean['date_past_1'].map(date_to_hr_resting_dict)
df_hr_mean['resting_min_hr_lag2'] = df_hr_mean['date_past_1'].map(date_to_hr_resting_minimum_dict)
df_hr_mean['resting_hr_lag3'] = df_hr_mean['date_past_2'].map(date_to_hr_resting_dict)
df_hr_mean['resting_min_hr_lag3'] = df_hr_mean['date_past_2'].map(date_to_hr_resting_minimum_dict)
df_hr_mean['resting_hr_lag4'] = df_hr_mean['date_past_3'].map(date_to_hr_resting_dict)
df_hr_mean['resting_min_hr_lag4'] = df_hr_mean['date_past_3'].map(date_to_hr_resting_minimum_dict)
df_hr_mean['resting_hr_lag5'] = df_hr_mean['date_past_4'].map(date_to_hr_resting_dict)
df_hr_mean['resting_min_hr_lag5'] = df_hr_mean['date_past_4'].map(date_to_hr_resting_minimum_dict)


# activity summary. not sure if this is to use or not?
df_activity_summary = pd.read_pickle('df_activity_summary.pkl')
# use this to get activity data into analytic file
df_activity = pd.read_pickle('df_activity.pkl')
# think i have this info below in dict form already
df_activity_summary.columns
# can use this to compute running and walking.
# hold off for now.
df_activity.columns
df_activity['activity'].value_counts(normalize=True)

# alcohol
with open('date_to_alcohol_dict.pkl', 'rb') as picklefile:
    date_to_alcohol_dict = pickle.load(picklefile)
with open('date_to_alcohol_type_dict.pkl', 'rb') as picklefile:
    date_to_alcohol_type_dict = pickle.load(picklefile)

df_hr_mean['alcohol_lag1'] = df_hr_mean['date_sleep'].map(date_to_alcohol_dict)
df_hr_mean[df_hr_mean['alcohol_lag1'].isnull()]
df_hr_mean['alcohol_lag2'] = df_hr_mean['date_past_1'].map(date_to_alcohol_dict)

# start of sleep time
with open('date_to_start_sleep_dict.pkl', 'rb') as picklefile:
    date_to_start_sleep_dict = pickle.load(picklefile)

df_hr_mean['start_sleep_time'] = df_hr_mean['date_sleep'].map(date_to_start_sleep_dict)
df_hr_mean['start_sleep_time_lag1'] = df_hr_mean['date_past_1'].map(date_to_start_sleep_dict)
len(df_hr_mean[df_hr_mean['start_sleep_time'].isnull()])  # 30

# minutes asleep 
with open('date_to_min_asleep_dict.pkl', 'rb') as picklefile:
    date_to_min_asleep_dict = pickle.load(picklefile)

df_hr_mean['minutes_asleep'] = df_hr_mean['date_sleep'].map(date_to_min_asleep_dict)
df_hr_mean['minutes_asleep_lag1'] = df_hr_mean['date_past_1'].map(date_to_min_asleep_dict)

# pickle time within sleep -- from start to end of sleep including times awake in between
with open('date_to_min_within_sleep_dict.pkl', 'rb') as picklefile:
    date_to_min_within_sleep_dict = pickle.load(picklefile)

df_hr_mean['minutes_asleep_alt'] = df_hr_mean['date_sleep'].map(date_to_min_within_sleep_dict)
df_hr_mean['minutes_asleep_alt_lag1'] = df_hr_mean['date_past_1'].map(date_to_min_within_sleep_dict)

# daily ratings
with open('date_to_subjective_sleep_dict.pkl', 'rb') as picklefile:
    date_to_subjective_sleep_dict = pickle.load(picklefile)
with open('date_to_fun_dict.pkl', 'rb') as picklefile:
    date_to_fun_dict = pickle.load(picklefile)
with open('date_to_energy_dict.pkl', 'rb') as picklefile:
    date_to_energy_dict = pickle.load(picklefile)

df_hr_mean['subjective_sleep_quality'] = df_hr_mean['date_sleep'].map(date_to_subjective_sleep_dict)
df_hr_mean['subjective_sleep_quality_lag1'] = df_hr_mean['date_past_1'].map(date_to_subjective_sleep_dict)

# types of sleep
with open('date_sleep_to_deep_minutes_dict.pkl', 'rb') as picklefile:
    date_sleep_to_deep_minutes_dict = pickle.load(picklefile)
with open('date_sleep_to_rem_minutes_dict.pkl', 'rb') as picklefile:
    date_sleep_to_rem_minutes_dict = pickle.load(picklefile)
with open('date_sleep_to_light_minutes_dict.pkl', 'rb') as picklefile:
    date_sleep_to_light_minutes_dict = pickle.load(picklefile)
with open('date_sleep_to_restless_minutes_dict.pkl', 'rb') as picklefile:
    date_sleep_to_restless_minutes_dict = pickle.load(picklefile)

df_hr_mean['deep_minutes'] = df_hr_mean['date_sleep'].map(date_sleep_to_deep_minutes_dict)
df_hr_mean['deep_minutes_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_deep_minutes_dict)
df_hr_mean['rem_minutes'] = df_hr_mean['date_sleep'].map(date_sleep_to_rem_minutes_dict)
df_hr_mean['rem_minutes_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_rem_minutes_dict)
df_hr_mean['light_minutes'] = df_hr_mean['date_sleep'].map(date_sleep_to_light_minutes_dict)
df_hr_mean['light_minutes_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_light_minutes_dict)
df_hr_mean['restless_minutes'] = df_hr_mean['date_sleep'].map(date_sleep_to_restless_minutes_dict)
df_hr_mean['restless_minutes_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_restless_minutes_dict)
# check missing -- lots of missing?

with open('date_sleep_to_hr_std_dict.pkl', 'rb') as picklefile:
    date_sleep_to_hr_std_dict = pickle.load(picklefile)
with open('date_sleep_to_deep_hr_dict.pkl', 'rb') as picklefile:
    date_sleep_to_deep_hr_dict = pickle.load(picklefile)
with open('date_sleep_to_deep_hr_std_dict.pkl', 'rb') as picklefile:
    date_sleep_to_deep_hr_std_dict = pickle.load(picklefile)
with open('date_sleep_to_deep_hr_ewm_from_wake_dict.pkl', 'rb') as picklefile:
    date_sleep_to_deep_hr_ewm_from_wake_dict = pickle.load(picklefile)
with open('date_sleep_to_deep_hr_ewm_from_onset_dict.pkl', 'rb') as picklefile:
    date_sleep_to_deep_hr_ewm_from_onset_dict = pickle.load(picklefile)

df_hr_mean['sleep_hr_std'] = df_hr_mean['date_sleep'].map(date_sleep_to_hr_std_dict)
df_hr_mean['sleep_hr_std_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_hr_std_dict)
df_hr_mean['deep_hr'] = df_hr_mean['date_sleep'].map(date_sleep_to_deep_hr_dict)
df_hr_mean['deep_hr_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_deep_hr_dict)
df_hr_mean['deep_hr_std'] = df_hr_mean['date_sleep'].map(date_sleep_to_deep_hr_std_dict)
df_hr_mean['deep_hr_std_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_deep_hr_std_dict)
df_hr_mean['deep_hr_near_wake'] = df_hr_mean['date_sleep'].map(date_sleep_to_deep_hr_ewm_from_wake_dict)
df_hr_mean['deep_hr_near_wake_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_deep_hr_ewm_from_wake_dict)
df_hr_mean['deep_hr_near_onset'] = df_hr_mean['date_sleep'].map(date_sleep_to_deep_hr_ewm_from_onset_dict)
df_hr_mean['deep_hr_near_onset_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_deep_hr_ewm_from_onset_dict)

with open('date_sleep_to_rem_hr_dict.pkl', 'rb') as picklefile:
    date_sleep_to_rem_hr_dict = pickle.load(picklefile)
with open('date_sleep_to_rem_hr_std_dict.pkl', 'rb') as picklefile:
    date_sleep_to_rem_hr_std_dict = pickle.load(picklefile)
with open('date_sleep_to_rem_hr_ewm_from_wake_dict.pkl', 'rb') as picklefile:
    date_sleep_to_rem_hr_ewm_from_wake_dict = pickle.load(picklefile)
with open('date_sleep_to_rem_hr_ewm_from_onset_dict.pkl', 'rb') as picklefile:
    date_sleep_to_rem_hr_ewm_from_onset_dict = pickle.load(picklefile)

df_hr_mean['rem_hr'] = df_hr_mean['date_sleep'].map(date_sleep_to_rem_hr_dict)
df_hr_mean['rem_hr_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_rem_hr_dict)
df_hr_mean['rem_hr_std'] = df_hr_mean['date_sleep'].map(date_sleep_to_rem_hr_std_dict)
df_hr_mean['rem_hr_std_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_rem_hr_std_dict)
df_hr_mean['rem_hr_near_wake'] = df_hr_mean['date_sleep'].map(date_sleep_to_rem_hr_ewm_from_wake_dict)
df_hr_mean['rem_hr_near_wake_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_rem_hr_ewm_from_wake_dict)
df_hr_mean['rem_hr_near_onset'] = df_hr_mean['date_sleep'].map(date_sleep_to_rem_hr_ewm_from_onset_dict)
df_hr_mean['rem_hr_near_onset_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_rem_hr_ewm_from_onset_dict)

with open('date_sleep_to_light_hr_dict.pkl', 'rb') as picklefile:
    date_sleep_to_light_hr_dict = pickle.load(picklefile)
with open('date_sleep_to_light_hr_std_dict.pkl', 'rb') as picklefile:
    date_sleep_to_light_hr_std_dict = pickle.load(picklefile)
with open('date_sleep_to_light_hr_ewm_from_wake_dict.pkl', 'rb') as picklefile:
    date_sleep_to_light_hr_ewm_from_wake_dict = pickle.load(picklefile)
with open('date_sleep_to_light_hr_ewm_from_onset_dict.pkl', 'rb') as picklefile:
    date_sleep_to_light_hr_ewm_from_onset_dict = pickle.load(picklefile)

df_hr_mean['light_hr'] = df_hr_mean['date_sleep'].map(date_sleep_to_light_hr_dict)
df_hr_mean['light_hr_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_light_hr_dict)
df_hr_mean['light_hr_std'] = df_hr_mean['date_sleep'].map(date_sleep_to_light_hr_std_dict)
df_hr_mean['light_hr_std_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_light_hr_std_dict)
df_hr_mean['light_hr_near_wake'] = df_hr_mean['date_sleep'].map(date_sleep_to_light_hr_ewm_from_wake_dict)
df_hr_mean['light_hr_near_wake_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_light_hr_ewm_from_wake_dict)
df_hr_mean['light_hr_near_onset'] = df_hr_mean['date_sleep'].map(date_sleep_to_light_hr_ewm_from_onset_dict)
df_hr_mean['light_hr_near_onset_lag1'] = df_hr_mean['date_past_1'].map(date_sleep_to_light_hr_ewm_from_onset_dict)

# hr at diff points in night
with open('date_to_span_from_wake_dict.pkl', 'rb') as picklefile:
    date_to_span_from_wake_dict = pickle.load(picklefile)
with open('date_to_halflife_from_wake_dict.pkl', 'rb') as picklefile:
    date_to_halflife_from_wake_dict = pickle.load(picklefile)
with open('date_to_span_from_onset_dict.pkl', 'rb') as picklefile:
    date_to_span_from_onset_dict = pickle.load(picklefile)
with open('date_to_halflife_from_onset_dict.pkl', 'rb') as picklefile:
    date_to_halflife_from_onset_dict = pickle.load(picklefile)

df_hr_mean['hr_near_wake_span'] = df_hr_mean['date_sleep'].map(date_to_span_from_wake_dict)
df_hr_mean['hr_near_wake_span_lag1'] = df_hr_mean['date_past_1'].map(date_to_span_from_wake_dict)
df_hr_mean['hr_near_wake_halflife'] = df_hr_mean['date_sleep'].map(date_to_halflife_from_wake_dict)
df_hr_mean['hr_near_wake_halflife_lag1'] = df_hr_mean['date_past_1'].map(date_to_halflife_from_wake_dict)
df_hr_mean['hr_near_sleep_onset_span'] = df_hr_mean['date_sleep'].map(date_to_span_from_onset_dict)
df_hr_mean['hr_near_sleep_onset_span_lag1'] = df_hr_mean['date_past_1'].map(date_to_span_from_onset_dict)
df_hr_mean['hr_near_sleep_onset_halflife'] = df_hr_mean['date_sleep'].map(date_to_halflife_from_onset_dict)
df_hr_mean['hr_near_sleep_onset_halflife_lag1'] = df_hr_mean['date_past_1'].map(date_to_halflife_from_onset_dict)

# activity
with open('date_to_distance_dict.pkl', 'rb') as picklefile:
    date_to_distance_dict = pickle.load(picklefile)
with open('date_to_elevation_dict.pkl', 'rb') as picklefile:
    date_to_elevation_dict = pickle.load(picklefile)
with open('date_to_steps_dict.pkl', 'rb') as picklefile:
    date_to_steps_dict = pickle.load(picklefile)
with open('date_to_active_minutes_dict.pkl', 'rb') as picklefile:
    date_to_active_minutes_dict = pickle.load(picklefile)

df_hr_mean['distance_lag1'] = df_hr_mean['date_sleep'].map(date_to_distance_dict)
df_hr_mean['distance_lag2'] = df_hr_mean['date_past_1'].map(date_to_distance_dict)
df_hr_mean['distance_lag3'] = df_hr_mean['date_past_2'].map(date_to_distance_dict)
df_hr_mean['elevation_lag1'] = df_hr_mean['date_sleep'].map(date_to_elevation_dict)
df_hr_mean['elevation_lag2'] = df_hr_mean['date_past_1'].map(date_to_elevation_dict)
df_hr_mean['elevation_lag3'] = df_hr_mean['date_past_3'].map(date_to_elevation_dict)
df_hr_mean['steps_lag1'] = df_hr_mean['date_sleep'].map(date_to_steps_dict)
df_hr_mean['steps_lag2'] = df_hr_mean['date_past_1'].map(date_to_steps_dict)
df_hr_mean['steps_lag3'] = df_hr_mean['date_past_2'].map(date_to_steps_dict)
df_hr_mean['active_lag1'] = df_hr_mean['date_sleep'].map(date_to_active_minutes_dict)
df_hr_mean['active_lag2'] = df_hr_mean['date_past_1'].map(date_to_active_minutes_dict)
df_hr_mean['active_lag3'] = df_hr_mean['date_past_3'].map(date_to_active_minutes_dict)


with open('date_to_steps_intensity_15_min_dict.pkl', 'rb') as picklefile:
    date_to_steps_intensity_15_min_dict = pickle.load(picklefile)
with open('date_to_steps_intensity_30_min_dict.pkl', 'rb') as picklefile:
    date_to_steps_intensity_30_min_dict = pickle.load(picklefile)
with open('date_to_steps_intensity_60_min_dict.pkl', 'rb') as picklefile:
    date_to_steps_intensity_60_min_dict = pickle.load(picklefile)
with open('date_to_floors_dict.pkl', 'rb') as picklefile:
    date_to_floors_dict = pickle.load(picklefile)

df_hr_mean['steps_intensity_15_lag1'] = df_hr_mean['date_sleep'].map(date_to_steps_intensity_15_min_dict)
df_hr_mean['steps_intensity_15_lag2'] = df_hr_mean['date_past_1'].map(date_to_steps_intensity_15_min_dict)
df_hr_mean['steps_intensity_15_lag3'] = df_hr_mean['date_past_2'].map(date_to_steps_intensity_15_min_dict)
df_hr_mean['steps_intensity_30_lag1'] = df_hr_mean['date_sleep'].map(date_to_steps_intensity_30_min_dict)
df_hr_mean['steps_intensity_30_lag2'] = df_hr_mean['date_past_1'].map(date_to_steps_intensity_30_min_dict)
df_hr_mean['steps_intensity_30_lag3'] = df_hr_mean['date_past_2'].map(date_to_steps_intensity_30_min_dict)
df_hr_mean['steps_intensity_60_lag1'] = df_hr_mean['date_sleep'].map(date_to_steps_intensity_60_min_dict)
df_hr_mean['steps_intensity_60_lag2'] = df_hr_mean['date_past_1'].map(date_to_steps_intensity_60_min_dict)
df_hr_mean['steps_intensity_60_lag3'] = df_hr_mean['date_past_2'].map(date_to_steps_intensity_60_min_dict)

df_hr_mean['floors_lag1'] = df_hr_mean['date_sleep'].map(date_to_floors_dict)
df_hr_mean['floors_lag2'] = df_hr_mean['date_past_1'].map(date_to_floors_dict)

df_hr_mean.tail()


# -----
# compute time-related confounds

# day of week
df_hr_mean['day_of_week'] = df_hr_mean['date_sleep'].dt.dayofweek
df_hr_mean['day_name'] = df_hr_mean['day_of_week'].replace([0,1,2,3,4,5,6], 
          ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])
df_hr_mean = pd.concat([df_hr_mean, pd.get_dummies(df_hr_mean['day_name'])], axis=1)

sns.barplot(x='day_name', y='hr_mean', data=df_hr_mean, 
            color='dodgerblue', alpha=.6, errcolor='grey', 
            order=['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])
plt.ylim(40,60)

df_hr_mean['weekend'] = 0
df_hr_mean.loc[(df_hr_mean['day_name']=='fri') | 
        (df_hr_mean['day_name']=='sat'), 'weekend'] = 1
df_hr_mean['weekend'].value_counts(normalize=True)

# season - monthly
# winter hr is higher than summer hr
# get avg temp of month. the high.
df_hr_mean['month'] = df_hr_mean['date_sleep'].dt.month
month_to_temp_dict = {1:39, 2:42, 3:50, 4:62, 5:72, 6:80, 
                      7:85, 8:84, 9:76, 10:65, 11:54, 12:44}
df_hr_mean['temp'] = df_hr_mean['month'].map(month_to_temp_dict)

sns.barplot(x='month', y='hr_mean', data=df_hr_mean, 
            color='dodgerblue', alpha=.6, errcolor='grey')
plt.ylim(40,60)

sns.relplot(x='temp', y='hr_mean', data=df_hr_mean, kind='line')
sns.lmplot(x='temp', y='hr_mean', data=df_hr_mean, 
           scatter_kws={'alpha':.15}, order=2)  
# probably order=2 is more reasonable  
# interesting that it's the sorta cold months that have highest hr
# but for the coldest, hr goes down again. though not enough data
# this could be due to many things other than weather/temperature.

# get general progressin of time with monthly count
df_hr_mean['year'] = df_hr_mean['date_sleep'].dt.year
df_hr_mean['month_ts'] = np.nan
df_month_ordered = df_hr_mean.groupby(['year', 'month']).size().reset_index()
df_month_ordered['year_month'] = df_month_ordered['year'].astype(str)+' '+df_month_ordered['month'].astype(str)
year_month_to_month_ordered_dict = dict(zip(df_month_ordered['year_month'], df_month_ordered.index))

df_hr_mean['year_month'] = df_hr_mean['year'].astype(str)+' '+df_hr_mean['month'].astype(str)
df_hr_mean['month_ts'] = df_hr_mean['year_month'].map(year_month_to_month_ordered_dict)

sns.lmplot(x='month_ts', y='hr_mean', data=df_hr_mean, scatter_kws={'alpha':.15})

results = smf.ols(formula = """hr_mean ~ month_ts""", data=df_hr_mean).fit()
print(results.summary())  # ns
# it's alcohol that makes signif (not any of the other variables)
results = smf.ols(formula = """hr_mean ~ month_ts + alcohol_lag1""", data=df_hr_mean).fit()
print(results.summary())  # p = .035 controlling for alcohol



# -----
# look at autocorr plots

variable = 'sleeping_hr'  # resting_hr  sleeping_hr

hr_series = pd.Series(df_hr_mean[variable].values)
hr_series.ffill(inplace=True)

autocorrelation_plot(hr_series)
plt.ylim(-.25, .40)
#plt.xlim(1,50)
plt.grid(False)
# resting hr - maybe lasting resting hr effects for 15-20 days?

# but really i want the partial autocorrelation: The partial 
# autocorrelation at lag k is the correlation that results 
# after removing the effect of any correlations due to the 
# terms at shorter lags. 
plot_pacf(hr_series, lags=10)  
# resting hr - 1 day back is the onely one very clearly correlated.
plot_pacf(hr_series, lags=20)  
# reting hr - though could also make the argument that up to about
# 10 days back, there's autocorrelation. then it really drops off.
# sleeping hr - again the prior night's sleeping hr is the most
# obviously corr. but could make arument that going back about
# 7 days there's autocorrelation.

# in general think these make arguments for including 1 lag
# and don't worry about including more unless some methodological
# reason, e.g., want hr resing prior to a hr-sleep lag variable.


# -----
# think about and create cumulative sleep q vars





# -----
# how to model past/lagged values with present if not linear




# -----
# anys

# take code from 1177 onwards in fitbit structure data for sleep 2

results = smf.ols(formula = """sleeping_hr ~ sleeping_hr_lag1 + 
                  sleeping_hr_lag2 + sleeping_hr_lag3""", data=df_hr_mean).fit()
print(results.summary())

results = smf.ols(formula = """resting_hr ~ resting_hr_lag1 + 
                  resting_hr_lag2 + resting_hr_lag3 + 
                  resting_hr_lag4""", data=df_hr_mean).fit()
print(results.summary())


results = smf.ols(formula = """resting_hr ~ sleeping_hr + resting_hr_lag1 + 
                  sleeping_hr_lag1 + resting_hr_lag2 + sleeping_hr_lag2""", data=df_hr_mean).fit()
print(results.summary())

results = smf.ols(formula = """resting_hr ~ sleeping_hr + resting_hr_lag1 + 
                  sleeping_hr_lag1 + temp + month_ts + start_sleep_time +
                  weekend + alcohol_lag1""", 
                  data=df_hr_mean).fit()
print(results.summary())
# strange that aclohol is neg corr with resting hr
# it's pos corr in straight corr. why would it flip signs?

df_hr_mean[['alcohol_lag1', 'resting_hr', 'sleeping_hr']].corr()
df_hr_mean[['start_sleep_time', 'fri', 'sat', 'wed']].corr()


# add activity and other variables
results = smf.ols(formula = """resting_hr ~ sleeping_hr + resting_hr_lag1 + 
                  sleeping_hr_lag1 + temp + month_ts + start_sleep_time +
                  weekend + alcohol_lag1 + hr_before_sleep_30_lag1 + 
                  steps_lag1 + elevation_lag1 + active_lag1 + 
                  steps_intensity_60_lag1""", 
                  data=df_hr_mean).fit()
print(results.summary())

df_hr_mean[['hr_before_sleep_30_lag1', 'hr_before_sleep_60_lag1', 
            'hr_before_sleep_120_lag1', 'resting_hr', 'sleeping_hr']].corr()

df_hr_mean[['distance_lag1', 'steps_lag1']].corr()
df_hr_mean[['elevation_lag1', 'floors_lag1']].corr()

df_hr_mean[['resting_hr', 'sleeping_hr', 'steps_lag1',
            'elevation_lag1', 'steps_intensity_60_lag1', 
            'steps_intensity_30_lag1', 'steps_intensity_15_lag1']].corr()

df_hr_mean[['resting_hr', 'sleeping_hr', 'steps_lag2',
            'elevation_lag2', 'steps_intensity_60_lag2']].corr()
# wow, activity totally reverses direction of relatinship at lag2

df_hr_mean[['resting_hr', 'sleeping_hr', 'steps_lag3',
            'elevation_lag3', 'steps_intensity_60_lag3']].corr()







