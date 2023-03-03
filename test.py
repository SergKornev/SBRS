import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import datetime
import calendar
from scipy import sparse
from pandas.api.types import CategoricalDtype
import time
from tqdm import tqdm

events_file = 'datasets/retailrocket/events.csv'
events_df = pd.read_csv(events_file)

events_actions = {
    'view': 1,
    'addtocart':2,
    'transaction':3
}
events_df.event = [events_actions[item] for item in events_df.event]

events_df['event_date'] = pd.to_datetime(events_df['timestamp'])
events_df = events_df.drop('timestamp', axis=1)

events_df['date'] = events_df.event_date.dt.date
events_df['time'] = events_df.event_date.dt.time
events_df['year'] = events_df.event_date.dt.year
events_df['month'] = events_df.event_date.dt.month
events_df['day'] = events_df.event_date.dt.day
events_df['hour'] = events_df.event_date.dt.hour
events_df['minute'] = events_df.event_date.dt.minute
events_df['sec'] = events_df.event_date.dt.second
events_df = events_df.drop(['event_date','time','date'], axis=1)

list_df = list()
for i in tqdm(range(len(events_df.visitorid.unique()))):
    time.sleep(0.000000001)
    list_df.append(events_df[events_df.visitorid == i].itemid.to_list())
print(list_df)

