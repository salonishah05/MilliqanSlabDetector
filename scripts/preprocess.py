import numpy as np
import pandas as pd 
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import os


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv("/data/1604All.csv")

# Apply cuts 
print(len(data))
data = data[data['risetime'] >= 3]
data = data[data['falltime'] >= 5]
print(len(data))
data = data[(data['premean'] <= 5) & (data['premean'] > -5)]
data = data[data['prerms'] <= 5]
data = data[(data['sidebandMean'] <= 5) & (data['sidebandMean'] >= -5)]
data = data[data['sidebandRMS'] <= 5]
print(len(data))
data = data[data['duration'] >= 50]
data = data[data["ipulse"] == 0]
print(len(data))

data = data.copy()
data['V_raw'] = data['V'].values
data['area_raw'] = data['area'].values

base_features = [
    'V', 'area', 'time', 'fittimeoffline', 'fittime', 'fitdtime',
    'halftime', 'fitslope', 'fitnpoints', 'duration',
    'risetime', 'falltime', 'premean', 'prerms',
    'sidebandMean', 'sidebandRMS', 'sidebandMeanRaw'
]

data['is_sat_region'] = (data['V_raw'] >= 500).astype('int8')
data['chan_is_oversat'] = data['chan'].isin([15, 10, 11, 2, 3]).astype('int8')
data['V_over_500'] = np.maximum(data['V_raw'] - 500.0, 0.0)

log_features = ["risetime", "sidebandRMS", "falltime", "area", "duration", "V"]


for col in log_features:
    data[col + "_log1p"] = np.log1p(np.clip(data[col].values, a_min=0, a_max=None))

numeric_cols = [
    'V', 'area', 'time',  'duration',
    'risetime', 'premean', 'prerms',
    'sidebandMean', 'sidebandRMS', 'sidebandMeanRaw',
    'is_sat_region', 'V_over_500', 'chan_is_oversat'
]

raw2log = {
    'V': 'V_log1p',
    'area': 'area_log1p',
    'risetime': 'risetime_log1p',
    'falltime': 'falltime_log1p',
    'duration': 'duration_log1p',
    'sidebandRMS': 'sidebandRMS_log1p'
}

numeric_cols = [raw2log.get(c, c) for c in numeric_cols]
model_df = data[numeric_cols + ['chan', 'EVT']].copy()

train_idx, val_idx = train_test_split(
    model_df.index, test_size=0.2, random_state=42, shuffle=True,
    stratify=model_df["chan"]
)

train_df = model_df.loc[train_idx].copy().reset_index(drop=True)
val_df   = model_df.loc[val_idx].copy().reset_index(drop=True)

chan_train = train_df["chan"].astype("int32").values
chan_val   = val_df["chan"].astype("int32").values

X_train_raw = train_df[numeric_cols].values.astype(np.float32)
X_val_raw   = val_df[numeric_cols].values.astype(np.float32)

global_scaler = RobustScaler()
X_train = global_scaler.fit_transform(X_train_raw).astype(np.float32)
X_val   = global_scaler.transform(X_val_raw).astype(np.float32)

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/global_robust_scaler.pkl", "wb") as f:
    pickle.dump({"numeric_cols": numeric_cols, "scaler": global_scaler}, f)

np.save("artifacts/X_train.npy", X_train)
np.save("artifacts/X_val.npy", X_val)
np.save("artifacts/chan_train.npy", chan_train)
np.save("artifacts/chan_val.npy", chan_val)

print("X_train:", X_train.shape, "X_val:", X_val.shape)
print("Unique channels in TRAIN:", np.unique(chan_train))
print("Unique channels in VAL:  ", np.unique(chan_val))

