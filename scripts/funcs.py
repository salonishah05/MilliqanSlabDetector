import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import numpy as np
import pickle
from pathlib import Path
from tensorflow import keras
import math
import pandas as pd

COLS = ['Run','EVT','chan','V','area','time','fittimeoffline','fittime','fitdtime','halftime',
        'fitslope','fitnpoints','fitprob','ipulse','duration','sidebandMean','sidebandRMS',
        'qual','risetime','falltime','premean','prerms','sidebandMeanRaw']


# function to ensure run number 
def ensureRuns(df):
    if df['Run'].nunique() == 1: 
        return True
    else: 
        print('All Run numbers are not the same. Here are all the run numbers present: ')
        print(df['Run'].unique(), end = "\n\n\n")
        return False

# reading data url based 
# for https://dstuart.physics.ucsb.edu/mq3/Run[]/MQRun[]_mqspulses.ant 
def readHtmlUrl2(file, max_rows=None, skip_rows=0):
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    try:
        response = requests.get(file, verify=False)
        response.raise_for_status()
        lines = response.text.splitlines()

        # NEW: slice lines before parsing
        if skip_rows:
            lines = lines[skip_rows:]
        if max_rows is not None:
            lines = lines[:max_rows]

        data = [line.split() for line in lines]
        df = pd.DataFrame(data)

        if len(df.columns) == 23:
            df.columns = COLS
            df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
            df.name = file[file.find('data'):file.find('.html')]
        else:
            print("Error: Column amount doesn't match")
            return None

        if ensureRuns(df):
            return df
        else:
            print("Error-> not all runs the same ")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {file}: {e}")
        return None

    

# Which channels are not present?
def missingChannels(df): 
    listChannels = df['chan'].unique()
    listChannels = np.sort(pd.to_numeric(listChannels))
    missing = np.setdiff1d(np.arange(0,80), listChannels)

    if missing.size > 0:
        print("All are not present. Missing channels: ")
        print(missing, end = "\n\n\n")
    else: 
        print("All are present", end = "\n\n\n")


def to_int(x):
    try:
        return int(float(x))
    except Exception:
        return pd.NA

def to_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return pd.NA
    except Exception:
        return pd.NA

CONVERTERS = {
    'Run': to_int, 'EVT': to_int, 'chan': to_int, 'ipulse': to_int, 'fitnpoints': to_int, 'qual': to_int,
    'V': to_float, 'area': to_float, 'time': to_float, 'fittimeoffline': to_float, 'fittime': to_float,
    'fitdtime': to_float, 'halftime': to_float, 'fitslope': to_float, 'fitprob': to_float, 'duration': to_float,
    'sidebandMean': to_float, 'sidebandRMS': to_float, 'risetime': to_float, 'falltime': to_float,
    'premean': to_float, 'prerms': to_float, 'sidebandMeanRaw': to_float
}

def ensure_runs_same(df):
    first = df['Run'].iloc[0]
    return (df['Run'] == first).all()

def apply_cuts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("Before cuts:", len(df))
    df = df[df['risetime'] >= 3]
    df = df[df['falltime'] >= 5]
    df = df[(df['premean'] <= 5) & (df['premean'] > -5)]
    df = df[df['prerms'] <= 5]
    df = df[(df['sidebandMean'] <= 5) & (df['sidebandMean'] >= -5)]
    df = df[df['sidebandRMS'] <= 5]
    df = df[df['duration'] >= 50]
    df = df[df["ipulse"] == 0]
    print("After cuts:", len(df))
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['V_raw'] = df['V'].values
    df['area_raw'] = df['area'].values

    df['is_sat_region'] = (df['V_raw'] >= 500).astype('int8')
    df['chan_is_oversat'] = df['chan'].isin([15, 10, 11, 2, 3]).astype('int8')
    df['V_over_500'] = np.maximum(df['V_raw'] - 500.0, 0.0)

    log_features = ["risetime", "sidebandRMS", "falltime", "area", "duration", "V"]

    for col in log_features:
        df[col + "_log1p"] = np.log1p(np.clip(df[col].values, a_min=0, a_max=None))

    return df

def transform_per_channel(df: pd.DataFrame, cols, scalers):
    X = np.empty((len(df), len(cols)), dtype=np.float32)
    for c, idxs in df.groupby('chan').groups.items():
        scaler = scalers[int(c)]
        X[df.index.get_indexer(idxs), :] = scaler.transform(df.loc[idxs, cols].values)
    return X

def preprocess_new_data(df_raw, scaler_artifact_path="artifacts/global_robust_scaler.pkl"):
    with open(scaler_artifact_path, "rb") as f:
        art = pickle.load(f)

    numeric_cols = art["numeric_cols"]
    scaler = art["scaler"]  

    df = apply_cuts(df_raw)
    df = add_derived_features(df)

    model_df = df[numeric_cols + ["chan"]].copy().reset_index(drop=True)

    chan_new = model_df["chan"].astype("int32").values
    meta_new = df[["EVT"]].copy().reset_index(drop=True) if "EVT" in df.columns else None

    X_raw = model_df[numeric_cols].values.astype(np.float32)
    X_new = scaler.transform(X_raw).astype(np.float32)

    return X_new, chan_new, meta_new, df.reset_index(drop=True), numeric_cols

# http://cms2.physics.ucsb.edu/mqslab/Run1481/MQSlabRun1481_mqspulses.ant
def rl(url, outputFile, pulse_output, model_path):
    # pull data off url
    newDf = readHtmlUrl2(url, max_rows = 6000000)
    print(newDf.head(10).round(2)['EVT'])

    # preprocess all data and cut 
    X_new, chan_new, meta_new, df_processed, num_cols = preprocess_new_data(newDf)

    # load model 
    model = keras.models.load_model(model_path)

    X_val_pred = model.predict({"features": X_new, "chan_id": chan_new}, batch_size=1024, verbose=1)
    val_mse = float(np.mean(np.mean((X_new - X_val_pred)**2, axis=1)))

    print(f"Mean reconstruction MSE on validation: {val_mse:.6f}")

    feature_cols = num_cols.copy()
    df_pulses = pd.DataFrame(X_new, columns=feature_cols, index=df_processed.index)


    err = X_new - X_val_pred
    RL = np.mean(err**2, axis=1)


    df_pulses["channel"] = chan_new
    df_pulses["RL"] = RL

    df_chan = compute_metrics(
        RL = RL,
        chan_new = chan_new.astype("int32")

    )


    if meta_new is not None and "EVT" in meta_new.columns:
        df_pulses["EVT"] = meta_new["EVT"].to_numpy()

    
    print(df_chan.head())
    append_metrics(df_chan, outputFile)
    append_pulse_metrics(df_pulses, pulse_output)
    print(df_chan.head(5))
    print(df_pulses.head(5))
    return df_chan, df_pulses

def compute_metrics(RL, chan_new, thr_path="artifacts/thr_by_chan.npy"):

    try:
        THR = np.load(thr_path)
        THR = np.asarray(THR).reshape(-1)
    except Exception:
        THR = None

    chans = np.unique(chan_new)
    rows = []


    for ch in chans:
        ch_int = int(ch)
        mask = (chan_new == ch)
        e = RL[mask]

        row = {
            "channel": ch_int,
            "mean_err": float(np.mean(e)) if len(e) else np.nan,
            "std_err": float(np.std(e)) if len(e) else np.nan,
            "n_samples": int(len(e)),
        }

        thr = None
        if THR is not None and 0 <= ch_int < len(THR):
            thr = float(THR[ch_int])
            if not np.isfinite(thr):
                thr = None

        if thr is not None and len(e):
            n_anom = int(np.sum(e > thr))
            frac_anom = float(n_anom / len(e))
        else:
            n_anom = 0
            frac_anom = 0.0
            thr = np.nan

        row.update({
            "thr": float(thr) if np.isfinite(thr) else np.nan,
            "n_anom": n_anom,
            "frac_anom": frac_anom,
        })

        rows.append(row)

    df = pd.DataFrame(rows)

    conditions = [
        df["frac_anom"] >= 0.01,
        df["frac_anom"] >= 0.03,
    ]
    choices = ["High", "Medium"]
    df["rating"] = np.select(conditions, choices, default="Low")

    return df

def append_metrics(df_new: pd.DataFrame, outputFile: str | Path):
    outputFile = Path(outputFile)
    outputFile.parent.mkdir(parents=True, exist_ok=True)

    if outputFile.suffix == "":
        outputFile = outputFile.with_suffix(".parquet")

    df_new.to_parquet(outputFile, index=False)
        
def append_pulse_metrics(df_new: pd.DataFrame, outputFile: str | Path):
    outputFile = Path(outputFile)
    outputFile.parent.mkdir(parents=True, exist_ok=True)

    if outputFile.suffix == "":
        outputFile = outputFile.with_suffix(".parquet")
    
    df_new.to_parquet(outputFile, index=False)