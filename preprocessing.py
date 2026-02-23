import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # for data normalization 
import joblib # to save the scaler object


# read csv data
df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")

# str to date_time object conversion
df["date_time"] = pd.to_datetime(df["date_time"])

# set date_time as index
df.set_index("date_time", inplace= True)

# time-based features
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month

# input variables
features = ["temp", "rain_1h", "snow_1h", "clouds_all", "hour", "dayofweek", "month"] # input features
# target features
target = "traffic_volume"

df = df[features + [target]].dropna() # take the necessary columns and remove null lines
print(df.head())

# normalization (between 0-1)
scaler_X = MinMaxScaler() # input scaler
scaler_y = MinMaxScaler() # target scaler

X_scaled = scaler_X.fit_transform(df[features]) # normalize input data
y_scaled = scaler_y.fit_transform(df[[target]])  # normalize target variable

# save scaler objects in order to use them in model prediction
joblib.dump(scaler_X, "scaler_X.save")
joblib.dump(scaler_y, "scaler_y.save")

# windowing: create sequence for time series
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i : i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LEN = 24 # sequence length 24 hours

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)
print(X_seq[: 1])
print(y_seq[: 1])

split_idx = int(0.8 * len(X_seq))

# training data
X_train = X_seq[:split_idx]
y_train = y_seq[:split_idx]

# test data
X_test = X_seq[split_idx:]
y_test = y_seq[split_idx:]

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Saving completed.")