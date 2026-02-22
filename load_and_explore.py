'''
Traffic volume forecasting with GRU (Gated Recurrent Unit)
'''

import pandas as pd # data analysis
import numpy as np  # math operations
import matplotlib.pyplot as plt # visualization
import seaborn as sns # advanced visualization


# load data
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
print(df.head()) # first 5 data line

# general info about the data
print(df.info())

# number of null values
print(df.isnull().sum())

# basic statistical summary
print(df.describe())

# organize time column
df["date_time"] = pd.to_datetime(df["date_time"]) # (str) date_time to (object) date_time
df.set_index("date_time", inplace=True) # date_time index
print(df.head())

# visualization
# drawing based on traffic volume 
plt.figure()
plt.plot(df["traffic_volume"], label = "Trafic Volume", color = "steelblue")
plt.title("Traffic Volume Time Series")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.legend()
plt.tight_layout()
plt.show()

# hourly average traffic volume
df["hour"] = df.index.hour # create hour data column

# group by hourly traffic volume
hourly_avg = df.groupby("hour")["traffic_volume"].mean()

# bar plot visualization, hourly, during the day
plt.figure()
sns.barplot(x = hourly_avg.index, y = hourly_avg.values, palette="viridis")
plt.title("Hourly Average Traffic Volume During the Day")
plt.xlabel("hour")
plt.ylabel("Average Traffic")
plt.show()