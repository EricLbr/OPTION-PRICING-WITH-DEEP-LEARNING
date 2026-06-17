# -*- coding: utf-8 -*-
"""
================================================================================
OPTION PRICING WITH DEEP LEARNING — Multi-Layer Perceptron (MLP) & Multi-Task Learning MODEL
================================================================================

Author: Eric Lambure

Master's Thesis: "To what extent can a deep neural network replicate or outperform traditional models in pricing options?"

Based on Ke & Yang (2019) — MLP2 architecture, extended with:
  - Volatility surface & per-stock skew features
  - CBOE SKEW index (market-wide tail risk)
  - Implied volatility from option chain
  - Liquidity features (volume, open interest, bid/ask sizes)

This is the first part of the code where the data is processed and clean before the training. 
Designed to run locally; Google Collab did not offered enough RAM for our dataset.
The training part, will be run on Collab in order to USE the T4 GPU for more computational power. 

If this code is to be used, the path pointing to the base files must be updated. 
The files are available upon request: eric.lambure.23@neoma-bs.com
================================================================================
"""
#%% CELL 1 — SETUP

import warnings
warnings.filterwarnings('ignore')

import os, gc, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


BASE_PATH        = 'C:/Users/Data/'
EQUITY_FILE      = os.path.join(BASE_PATH, 'Equity/equity_daily.parquet')
RF_FILE          = os.path.join(BASE_PATH, 'Rates/risk_free_rates.parquet')
SKEW_FILE        = os.path.join(BASE_PATH, 'Skew/SKEW.parquet')
VOL_SURFACE_DIR  = os.path.join(BASE_PATH, 'Vol_Surface/')
OPTION_CHAIN_DIR = os.path.join(BASE_PATH, 'Options/Option_dataset')
MODEL_SAVE_DIR   = os.path.join(BASE_PATH, 'models/')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

TICKERS = [
    'AAPL','ABBV','AMD','AMZN','AVGO','BAC','CAT','COST','CSCO','CVX',
    'GE','GOOGL','GS','HD','IWM','JNJ','JPM','KO','LLY','MA',
    'META','MRK','MS','MSFT','NFLX','NVDA','ORCL','PG','QQQ','RTX',
    'SCHW','SPY','TSLA','UNH','V','WFC','WMT','XOM','DIS'
]

#%% CELL 2 — LOAD EQUITY + RISK-FREE + SKEW + VOL SURFACE

# Equity data
equity_df = pd.read_parquet(EQUITY_FILE, engine='fastparquet')
equity_df['date'] = pd.to_datetime(equity_df['date'])
for c in ['PX_LAST','realized_vol_20d']:
    equity_df[c] = pd.to_numeric(equity_df[c], errors='coerce')
equity_df = equity_df[['ticker','date','PX_LAST','realized_vol_20d']].copy()
equity_df.drop_duplicates(subset=['ticker','date'], inplace=True)
print(f"Equity: {len(equity_df):,} rows")

# Risk-free rates
rf_df = pd.read_parquet(RF_FILE, engine='fastparquet')
rf_df = rf_df.reset_index()
rf_df['date'] = pd.to_datetime(rf_df['date'])
for c in ['T_3M','T_6M','T_2Y','SOFR']:
    rf_df[c] = pd.to_numeric(rf_df[c], errors='coerce')
if rf_df[['T_3M','T_6M','T_2Y']].median().median() > 1:
    for c in ['T_3M','T_6M','T_2Y','SOFR']:
        rf_df[c] = rf_df[c] / 100.0
rf_df = rf_df[['date','T_3M','T_6M','T_2Y','SOFR']].drop_duplicates(subset=['date'])
print(f"Risk-free: {len(rf_df):,} rows")

# CBOE SKEW index
skew_df = pd.read_parquet(SKEW_FILE, engine='fastparquet')
skew_df = skew_df.reset_index()
skew_df['date'] = pd.to_datetime(skew_df['date'])
skew_df['SKEW'] = pd.to_numeric(skew_df['SKEW'], errors='coerce')
# Normalize to make it easier for the network: center around 100, scale down
skew_df['skew_index'] = (skew_df['SKEW'] - 100.0) / 100.0
skew_df = skew_df[['date','skew_index']].drop_duplicates(subset=['date'])
print(f"CBOE SKEW: {len(skew_df):,} rows "
      f"(raw range: {skew_df['skew_index'].min()*100+100:.0f} – "
      f"{skew_df['skew_index'].max()*100+100:.0f})")

# Volatility surface
vol_frames = []
for ticker in TICKERS:
    fpath = os.path.join(VOL_SURFACE_DIR, f'{ticker}_vol_surface.parquet')
    tmp = pd.read_parquet(fpath, engine='fastparquet')
    if 'ticker' not in tmp.columns:
        tmp['ticker'] = ticker
    vol_frames.append(tmp)
vol_surf_df = pd.concat(vol_frames, ignore_index=True)
del vol_frames; gc.collect()
vol_surf_df['date'] = pd.to_datetime(vol_surf_df['date'])
for c in ['iv_atm_30d','iv_atm_60d','iv_90m_30d','iv_110m_30d']:
    vol_surf_df[c] = pd.to_numeric(vol_surf_df[c], errors='coerce')
# Per-stock skew: OTM put vol minus OTM call vol (micro/individual tail risk)
vol_surf_df['skew_30d'] = vol_surf_df['iv_90m_30d'] - vol_surf_df['iv_110m_30d']
vol_surf_df = vol_surf_df[['ticker','date','iv_atm_30d','iv_atm_60d',
                            'iv_90m_30d','iv_110m_30d','skew_30d']].copy()
vol_surf_df.drop_duplicates(subset=['ticker','date'], inplace=True)
print(f"Vol surface: {len(vol_surf_df):,} rows, per-stock skew_30d computed")

#%% CELL 3 — OPTION CHAIN PROCESSING

OPTION_COLS_KEEP = [
    'strike', 'type', 'bid', 'bid_size', 'ask', 'ask_size',
    'volume', 'open_interest', 'date', 'implied_volatility',
    'expiration', 'days_to_expiry'
]

TENORS_YEARS = np.array([0.25, 0.5, 2.0])
TENOR_COLS   = ['T_3M', 'T_6M', 'T_2Y']

FEATURE_COLS = [
    'S', 'K', 'T', 'r', 'sigma',
    'moneyness', 'log_moneyness', 'implied_volatility',
    'iv_atm_30d', 'iv_atm_60d', 'iv_90m_30d', 'iv_110m_30d',
    'skew_30d', 'skew_index',
    'volume', 'open_interest', 'bid_size', 'ask_size',
]

def match_rf_rate(T_series, rf_row_df):
    T_vals = T_series.values.reshape(-1, 1)
    closest_idx = np.argmin(np.abs(T_vals - TENORS_YEARS), axis=1)
    rf_vals = rf_row_df[TENOR_COLS].values
    return pd.Series(rf_vals[np.arange(len(closest_idx)), closest_idx], index=T_series.index)


def process_single_ticker(ticker):
    fpath = os.path.join(OPTION_CHAIN_DIR, f'{ticker.lower()}_options.parquet')
    df = pd.read_parquet(fpath, columns=OPTION_COLS_KEEP, engine='fastparquet')

    df['date'] = pd.to_datetime(df['date'])
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['ticker'] = ticker

    if 'days_to_expiry' in df.columns:
        df['T'] = pd.to_numeric(df['days_to_expiry'], errors='coerce') / 365.25
    else:
        df['T'] = (df['expiration'] - df['date']).dt.days / 365.25

    for c in ['strike','bid','bid_size','ask','ask_size','volume','open_interest','implied_volatility']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Filter early
    df = df[(df['T'] > 0) & (df['T'] <= 3.0)].copy()
    df = df[(df['bid'] >= 0) & (df['ask'] > 0)].copy()
    df.dropna(subset=['strike','bid','ask','T','implied_volatility'], inplace=True)

    # Merge equity
    eq = equity_df[equity_df['ticker'] == ticker]
    df = df.merge(eq[['date','PX_LAST','realized_vol_20d']], on='date', how='left')
    df.rename(columns={'PX_LAST': 'S'}, inplace=True)
    df.dropna(subset=['S'], inplace=True)

    # Merge risk-free
    df = df.merge(rf_df, on='date', how='left')
    df['r'] = match_rf_rate(df['T'], df)
    df.drop(columns=['T_3M','T_6M','T_2Y','SOFR'], inplace=True)
    df.dropna(subset=['r'], inplace=True)

    # Merge volatility surface
    vs = vol_surf_df[vol_surf_df['ticker'] == ticker]
    df = df.merge(vs.drop(columns='ticker'), on='date', how='left')

    # Merge CBOE SKEW
    df = df.merge(skew_df, on='date', how='left')

    # Feature engineering
    df.rename(columns={'strike': 'K'}, inplace=True)
    df['moneyness']     = df['S'] / df['K']
    df['log_moneyness'] = np.log(df['moneyness'])

    df['sigma'] = df['realized_vol_20d']
    mask_na = df['sigma'].isna()
    df.loc[mask_na, 'sigma'] = df.loc[mask_na, 'implied_volatility']

    df['is_call'] = df['type'].astype(str).str.lower().str.strip().isin(['call','c','calls']).astype(np.int8)

    df = df[(df['moneyness'] > 0.5) & (df['moneyness'] < 2.0)].copy()

    # --- MEMORY OPTIMIZATION: Clean inside the loop ---
    keep_cols = FEATURE_COLS + ['bid', 'ask', 'is_call', 'ticker', 'date']
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    for c in FEATURE_COLS:
        if c not in ['S','K','T','r','sigma'] and c in df.columns:
            df[c] = df[c].fillna(0.0)

    df.dropna(subset=['S','K','T','r','sigma','bid','ask'], inplace=True)

    float_cols = df.select_dtypes(include='float64').columns
    df[float_cols] = df[float_cols].astype(np.float32)

    return df

print("Processing option chains ticker by ticker...")
calls_list = []
puts_list = []

for i, ticker in enumerate(TICKERS):
    chunk = process_single_ticker(ticker)

    # Split calls and puts immediately to avoid building one massive dataframe
    calls_list.append(chunk[chunk['is_call'] == 1])
    puts_list.append(chunk[chunk['is_call'] == 0])

    del chunk
    gc.collect()

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(TICKERS)} done...")

# DELETE SOURCE DATAFRAMES BEFORE CONCATENATING (Memory Optimization)
print("Freeing memory from source tables...")
del equity_df, rf_df, skew_df, vol_surf_df
gc.collect()

print("Concatenating Calls...")
calls_df = pd.concat(calls_list, ignore_index=True)
del calls_list  # Free memory immediately
gc.collect()

print("Concatenating Puts...")
puts_df = pd.concat(puts_list, ignore_index=True)
del puts_list   # Free memory immediately
gc.collect()

print(f"\nFinal dataset: {len(calls_df):,} calls, {len(puts_df):,} puts.")
print(f"Calls RAM: {calls_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print(f"Puts RAM:  {puts_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

#%% CELL 4 — PREPARE TRAINING DATA

def prepare_split(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    Y = df[['bid','ask']].values.astype(np.float32)

    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    n_test = max(int(0.01 * n), 1)
    n_val  = max(int(0.01 * n), 1)
    n_train = n - n_test - n_val

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[idx[:n_train]])
    X_val   = scaler.transform(X[idx[n_train:n_train+n_val]])
    X_test  = scaler.transform(X[idx[n_train+n_val:]])

    return {
        'X_train': X_train, 'Y_train': Y[idx[:n_train]],
        'X_val': X_val,     'Y_val': Y[idx[n_train:n_train+n_val]],
        'X_test': X_test,   'Y_test': Y[idx[n_train+n_val:]],
        'scaler': scaler,
        'test_df': df.iloc[idx[n_train+n_val:]].reset_index(drop=True),
        'n_train': n_train, 'n_val': n_val, 'n_test': n - n_train - n_val
    }

data_splits = {}

# Process Calls
print("Scaling and splitting CALLS...")
data_splits['call'] = prepare_split(calls_df, FEATURE_COLS)
s = data_splits['call']
print(f"CALL: train={s['n_train']:,}  val={s['n_val']:,}  test={s['n_test']:,}")
del calls_df
gc.collect()

# Process Puts
print("Scaling and splitting PUTS...")
data_splits['put'] = prepare_split(puts_df, FEATURE_COLS)
s = data_splits['put']
print(f"PUT: train={s['n_train']:,}  val={s['n_val']:,}  test={s['n_test']:,}")
del puts_df
gc.collect()

print("\nReady for GPU training!")

#%% CELL 5 — SAVE PROCESSED DATA FOR COLAB

import joblib

print("\nSaving processed data splits to disk...")
print("Please wait, this might take a few minutes as the file will be quite large...")

# Define the exact save path using the MODEL_SAVE_DIR you defined in Cell 1
save_path = os.path.join(MODEL_SAVE_DIR, 'processed_data_splits.joblib')

# Dump the entire data_splits dictionary into the file
joblib.dump(data_splits, save_path, compress=3)

print(f"✓ Data successfully saved to: {save_path}")
print("You can now close Spyder, upload this file to your Google Drive, and switch to Colab!")
