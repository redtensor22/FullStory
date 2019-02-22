# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:53:34 2019

@author: Aaron
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import stats




df = pd.read_csv('C:/Users/Aaron/Desktop/Python Files/FullStory/yellow_tripdata_2017-06.csv')

df_lookup = pd.read_csv('C:/Users/Aaron/Desktop/Python Files/FullStory/taxi+_zone_lookup.csv')

print(df.dtypes)

#check for missing values
print(df.isnull().sum())

#no missing values here, can move on
#look at description of each columns
print(df.describe())

#there are clearly some bad values. Will handle relevant columns in each section

#we have some datetime columns we need to deal with 
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])


#PART 1
###########################
#lets find out the hour with the highest percentage tips
#naively start off just by looking at pick up hours with no location data
#for memory allocation will make new dataset only of needed columns
#############################

df_tip_hour_1 = df[['tpep_pickup_datetime','payment_type','total_amount','tip_amount']]

#some weird values. Lets drop outliers
q = df_tip_hour_1['total_amount'].quantile(0.99)

q2 = df_tip_hour_1['total_amount'].quantile(0.05)

df_tip_hour_1=df_tip_hour_1[df_tip_hour_1['total_amount'] < q]

df_tip_hour_1=df_tip_hour_1[df_tip_hour_1['total_amount'] > q2]

q = df_tip_hour_1['tip_amount'].quantile(0.99)

q2 = df_tip_hour_1['tip_amount'].quantile(0.05)

df_tip_hour_1=df_tip_hour_1[df_tip_hour_1['tip_amount'] < q]

df_tip_hour_1=df_tip_hour_1[df_tip_hour_1['tip_amount'] > q2]

#need to only look at values where credit was used (only ones tip is logged)
df_tip_hour_1 = df_tip_hour_1[df_tip_hour_1['payment_type']==1]

#create column with tip percentage 
df_tip_hour_1['Percentage'] = df_tip_hour_1['tip_amount']/(df_tip_hour_1['total_amount'])

df_tip_hour_1=df_tip_hour_1.dropna()

#look at only hours of the day in the time column
df_tip_hour_1['Hour']=df_tip_hour_1['tpep_pickup_datetime'].dt.hour

tips = df_tip_hour_1.groupby(['Hour']).mean()
#now average all the percentages for each hour
print('the hour with the best tip percentage is', (df_tip_hour_1.groupby(['Hour']).mean())['Percentage'].idxmax())
#do a bar chart of the tim percetage per hour
ax = tips.plot.bar(y='Percentage', color='darkblue', legend=False, rot=0)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Tip Percentage')


#PART 2
###################################
#best hour is 10am. Now lets see if the fare rates are the same for each hour, or what the 
#hour with the highest fare rate it
####################################
# first check how the fare values and distances look 
df_fare_rate = df[['trip_distance','fare_amount','tpep_pickup_datetime']]
print(df_fare_rate.describe())

#some weird values. Lets drop outliers
q = df_fare_rate['fare_amount'].quantile(0.99)

q2 = df_fare_rate['fare_amount'].quantile(0.05)

df_fare_rate=df_fare_rate[df_fare_rate['fare_amount'] < q]

df_fare_rate=df_fare_rate[df_fare_rate['fare_amount'] > q2]


df_fare_rate['Hour']=df_fare_rate['tpep_pickup_datetime'].dt.hour
fare_rate = df_fare_rate.groupby(['Hour']).mean()
fare_rate['rate'] = fare_rate['fare_amount']/fare_rate['trip_distance']
print('the hour with the best fare rate is',fare_rate['rate'].idxmax())

ax2 = fare_rate.plot.bar(y='rate', color='darkblue', legend=False, rot=0)
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Fare Rate')

#PART 3
###################################
#now look at highest density of pick
#ups per hour, and places with most passengers per hour
####################################
#first break into locations
#will consider bouroughs 

#first get new dataframe of data we will need
df_bouroughs = df[['tpep_pickup_datetime','passenger_count','PULocationID']]

#now merge the dictionary and new dataframe to get the boroughs
df_bouroughs_map = (df_bouroughs.merge(df_lookup, left_on='PULocationID', right_on='LocationID')
          .reindex(columns=['tpep_pickup_datetime', 'passenger_count', 'PULocationID', 'Borough']))

#some of the borough values are "unknown", will drop these
df_bouroughs_map = df_bouroughs_map[df_bouroughs_map['Borough']!='Unknown']

#now make a column of the hours
df_bouroughs_map['Hour']=df_bouroughs_map['tpep_pickup_datetime'].dt.hour

#now within each borough we will look at average passengers per hour
pass_per_hour = df_bouroughs_map.groupby(['Borough', 'Hour'], as_index = False).agg({'passenger_count': ['mean']})

#now for each hour we will find the borough with the highest passenger count
hour_best=df_bouroughs_map.groupby(['Hour','Borough'], as_index = False).agg({'passenger_count':'mean'})
print(hour_best.sort_values('Hour').drop_duplicates('Hour',keep='last'))

best_hour_df = hour_best.sort_values('Hour').drop_duplicates('Hour',keep='last')

#PART 4
###################################
#lets try to make a map visualization
#of the best taxi boroughs per time
####################################

import geopandas as gpd
import pandas as pd

# set the filepath and load in a shapefile
map_df = gpd.read_file('C:/Users/Aaron/Desktop/Python Files/FullStory/taxi_zones.shp')

map_df.head()
map_df.plot()

##
#would be a significant amount of code for a prelim project. 
#Idea is to have a scroll bar allowing one to change times of the 
#day and the borough of the map that is most profitable to 
#be highlighted  
##













