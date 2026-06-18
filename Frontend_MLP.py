# -*- coding: utf-8 -*-
"""
================================================================================
OPTION PRICING ENGINE — DESKTOP INTERFACE
================================================================================

Author: Eric Lambure

This app wraps the OptionPricingEngine from "MLP_Option_Pricing_Training.ipynb"
(Cell 13) in a desktop GUI allowing us to price options interactively.

It loads:
  - the trained MLP2 PyTorch models  (mlp2_call_final.pt / mlp2_put_final.pt)
  - the feature scalers used at training time
  - the feature column list (from config.json, if available)
  
If this code is to be used, the path pointing to the base files must be updated. 
The files are available upon request: eric.lambure.23@neoma-bs.com

================================================================================
"""

import os
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from scipy.stats import norm

import joblib
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


DEFAULT_BASE_PATH = "C:/Users/ericl/OneDrive/Bureau/ECOLE/Cours/M2/Thesis/Data"
MODELS_SUBDIR        = "Results"
DATA_SPLITS_SUBPATH  = os.path.join("models", "processed_data_splits.joblib")
SCALER_CACHE_NAME    = "scalers_cache.joblib"
CONFIG_NAME          = "config.json"

# Last-used base path between runs
APP_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".option_pricing_app_settings.json")

DEFAULT_FEATURE_COLS = [
    'S', 'K', 'T', 'r', 'sigma',
    'moneyness', 'log_moneyness', 'implied_volatility',
    'iv_atm_30d', 'iv_atm_60d', 'iv_90m_30d', 'iv_110m_30d',
    'skew_30d', 'skew_index',
    'volume', 'open_interest', 'bid_size', 'ask_size',
]


# ==============================================================================
# MLP2 MODEL (Cell 5 from the training notebook)
# ==============================================================================
class MLP2(nn.Module):
    """
    Ke & Yang (2019) MLP2:
      Input -> [Linear(400) + BN + LeakyReLU] x 3 -> Linear(2) + ReLU
    Output: (bid, ask).
    """
    def __init__(self, input_dim, hidden=[400, 400, 400]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.01)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.relu(self.head(self.backbone(x)))


# ==============================================================================
# TRADITIONAL PRICING MODELS — copied as-is from the training notebook (Cell 4)
# ==============================================================================
def black_scholes(S, K, T, r, sigma, option_type='call'):
    S, K, T, r, sigma = (np.asarray(x, dtype=np.float64) for x in [S, K, T, r, sigma])
    scalar_input = S.ndim == 0
    S, K, T, r, sigma = (np.atleast_1d(x) for x in [S, K, T, r, sigma])

    price = np.where(option_type == 'call',
                      np.maximum(S - K, 0.0),
                      np.maximum(K - S, 0.0)).astype(np.float64)
    valid = (T > 0) & (sigma > 0)
    if valid.any():
        Sv, Kv, Tv, rv, sv = S[valid], K[valid], T[valid], r[valid], sigma[valid]
        d1 = (np.log(Sv / Kv) + (rv + 0.5 * sv**2) * Tv) / (sv * np.sqrt(Tv))
        d2 = d1 - sv * np.sqrt(Tv)
        if option_type == 'call':
            price[valid] = Sv * norm.cdf(d1) - Kv * np.exp(-rv * Tv) * norm.cdf(d2)
        else:
            price[valid] = Kv * np.exp(-rv * Tv) * norm.cdf(-d2) - Sv * norm.cdf(-d1)

    return float(price[0]) if scalar_input else price


def binomial_tree(S, K, T, r, sigma, option_type='call', american=False, N=200):
    S, K, T, r, sigma = (np.asarray(x, dtype=np.float64) for x in [S, K, T, r, sigma])
    scalar_input = S.ndim == 0
    S, K, T, r, sigma = (np.atleast_1d(x) for x in [S, K, T, r, sigma])

    price = np.where(option_type == 'call',
                      np.maximum(S - K, 0.0),
                      np.maximum(K - S, 0.0)).astype(np.float64)
    valid = (T > 0) & (sigma > 0)
    if not valid.any():
        return float(price[0]) if scalar_input else price

    Sv, Kv, Tv, rv, sv = S[valid], K[valid], T[valid], r[valid], sigma[valid]

    dt   = Tv / N
    u    = np.exp(sv * np.sqrt(dt))
    d    = 1.0 / u
    p    = (np.exp(rv * dt) - d) / (u - d)
    disc = np.exp(-rv * dt)
    j  = np.arange(N + 1)
    ST = Sv[:, None] * u[:, None] ** (2 * j - N)[None, :]

    V = (np.maximum(ST - Kv[:, None], 0.0) if option_type == 'call'
         else np.maximum(Kv[:, None] - ST, 0.0))

    for i in range(N - 1, -1, -1):
        V = disc[:, None] * (p[:, None] * V[:, 1:] + (1 - p[:, None]) * V[:, :-1])
        if american:
            j_i  = np.arange(i + 1)
            ST_i = Sv[:, None] * u[:, None] ** (2 * j_i - i)[None, :]
            intrinsic = (np.maximum(ST_i - Kv[:, None], 0.0) if option_type == 'call'
                         else np.maximum(Kv[:, None] - ST_i, 0.0))
            V = np.maximum(V, intrinsic)

    price[valid] = V[:, 0]
    return float(price[0]) if scalar_input else price


def monte_carlo(S, K, T, r, sigma, option_type='call',
                 n_sims=5000, n_steps=1, seed=42, chunk_size=10_000):
    S, K, T, r, sigma = (np.asarray(x, dtype=np.float64) for x in [S, K, T, r, sigma])
    scalar_input = S.ndim == 0
    S, K, T, r, sigma = (np.atleast_1d(x) for x in [S, K, T, r, sigma])

    price = np.where(option_type == 'call',
                      np.maximum(S - K, 0.0),
                      np.maximum(K - S, 0.0)).astype(np.float64)
    valid = (T > 0) & (sigma > 0)
    if not valid.any():
        return float(price[0]) if scalar_input else price

    Sv, Kv, Tv, rv, sv = S[valid], K[valid], T[valid], r[valid], sigma[valid]
    n    = len(Sv)
    rng  = np.random.RandomState(seed)
    half = n_sims // 2
    price_valid = np.zeros(n)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Sc, Kc, Tc, rc, sc = Sv[start:end], Kv[start:end], Tv[start:end], rv[start:end], sv[start:end]
        nc = end - start

        log_ST = np.zeros((nc, n_sims))
        dt = Tc / n_steps
        for _ in range(n_steps):
            Z      = rng.standard_normal((nc, half))
            Z      = np.concatenate([Z, -Z], axis=1)
            log_ST += ((rc[:, None] - 0.5 * sc[:, None]**2) * dt[:, None]
                       + sc[:, None] * np.sqrt(dt[:, None]) * Z)

        ST = np.exp(np.log(Sc[:, None]) + log_ST)
        payoff = (np.maximum(ST - Kc[:, None], 0.0) if option_type == 'call'
                  else np.maximum(Kc[:, None] - ST, 0.0))
        price_valid[start:end] = np.exp(-rc * Tc) * payoff.mean(axis=1)
        del Z, log_ST, ST, payoff

    price[valid] = price_valid
    return float(price[0]) if scalar_input else price


def heston_mc(S, K, T, r, sigma, option_type='call',
              kappa=2.0, theta=None, xi=0.3, rho=-0.7,
              n_sims=2000, n_steps=50, seed=42, chunk_size=250):
    S, K, T, r, sigma = (np.asarray(x, dtype=np.float64) for x in [S, K, T, r, sigma])
    scalar_input = S.ndim == 0
    S, K, T, r, sigma = (np.atleast_1d(x) for x in [S, K, T, r, sigma])

    price = np.where(option_type == 'call',
                      np.maximum(S - K, 0.0),
                      np.maximum(K - S, 0.0)).astype(np.float64)
    valid = (T > 0) & (sigma > 0)
    if not valid.any():
        return float(price[0]) if scalar_input else price

    Sv, Kv, Tv, rv, sv = S[valid], K[valid], T[valid], r[valid], sigma[valid]
    n           = len(Sv)
    rng         = np.random.RandomState(seed)
    price_valid = np.zeros(n)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Sc, Kc, Tc, rc, sc = Sv[start:end], Kv[start:end], Tv[start:end], rv[start:end], sv[start:end]
        nc = end - start

        th = sc**2 if theta is None else np.full(nc, float(theta))
        v0 = sc**2
        dt = Tc / n_steps

        Z1 = rng.standard_normal((n_steps, nc, n_sims))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * rng.standard_normal((n_steps, nc, n_sims))

        S_p = np.broadcast_to(Sc[:, None], (nc, n_sims)).copy()
        v_p = np.broadcast_to(v0[:, None], (nc, n_sims)).copy()

        for t in range(n_steps):
            v_p  = np.maximum(v_p, 0.0)
            sv_p = np.sqrt(v_p)
            dts  = dt[:, None]
            S_p *= np.exp((rc[:, None] - 0.5 * v_p) * dts
                          + sv_p * np.sqrt(dts) * Z1[t])
            v_p += (kappa * (th[:, None] - v_p) * dts
                    + xi * sv_p * np.sqrt(dts) * Z2[t])

        payoff = (np.maximum(S_p - Kc[:, None], 0.0) if option_type == 'call'
                  else np.maximum(Kc[:, None] - S_p, 0.0))
        price_valid[start:end] = np.exp(-rc * Tc) * payoff.mean(axis=1)
        del Z1, Z2, S_p, v_p, sv_p, payoff

    price[valid] = price_valid
    return float(price[0]) if scalar_input else price


# ==============================================================================
# PRICING ENGINE
# ==============================================================================
class OptionPricingEngine:
    def __init__(self, models, scalers, feature_cols, device):
        self.models = models
        self.scalers = scalers
        self.fcols = feature_cols
        self.device = device

    def price(self, S, K, T, r, sigma, option_type='call',
              exercise='european',
              implied_vol=None,
              iv_atm_30d=None, iv_atm_60d=None,
              iv_90m_30d=None, iv_110m_30d=None,
              skew_30d=None, skew_index=None,
              volume=0, open_interest=0, bid_size=0, ask_size=0):

        ot = option_type.lower()
        am = exercise.lower() == 'american'
        if implied_vol is None:
            implied_vol = sigma
        if skew_30d is None and iv_90m_30d and iv_110m_30d:
            skew_30d = iv_90m_30d - iv_110m_30d

        res = {'inputs': dict(S=S, K=K, T=T, r=r, sigma=sigma,
                               option_type=ot, exercise=exercise)}

        t0 = time.time(); res['BlackScholes'] = black_scholes(S, K, T, r, sigma, ot)
        res['BS_ms'] = (time.time() - t0) * 1000
        t0 = time.time(); res['Binomial'] = binomial_tree(S, K, T, r, sigma, ot, am)
        res['Binom_ms'] = (time.time() - t0) * 1000
        t0 = time.time(); res['MonteCarlo'] = monte_carlo(S, K, T, r, sigma, ot, 100000)
        res['MC_ms'] = (time.time() - t0) * 1000
        t0 = time.time(); res['Heston'] = heston_mc(S, K, T, r, sigma, ot, n_sims=100000)
        res['Heston_ms'] = (time.time() - t0) * 1000

        if ot in self.models:
            t0 = time.time()
            feat_map = {
                'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
                'moneyness': S / K, 'log_moneyness': np.log(S / K),
                'implied_volatility': implied_vol,
                'iv_atm_30d': iv_atm_30d or sigma,
                'iv_atm_60d': iv_atm_60d or sigma,
                'iv_90m_30d': iv_90m_30d or sigma,
                'iv_110m_30d': iv_110m_30d or sigma,
                'skew_30d': skew_30d or 0.0,
                'skew_index': skew_index or 0.0,
                'volume': volume, 'open_interest': open_interest,
                'bid_size': bid_size, 'ask_size': ask_size,
            }
            x = np.array([[feat_map[c] for c in self.fcols]], dtype=np.float32)
            x = self.scalers[ot].transform(x)
            self.models[ot].eval()
            with torch.no_grad():
                p = self.models[ot](torch.tensor(x).to(self.device)).cpu().numpy()[0]
            res['MLP2_bid'] = float(p[0])
            res['MLP2_ask'] = float(p[1])
            res['MLP2'] = float((p[0] + p[1]) / 2)
            res['MLP2_ms'] = (time.time() - t0) * 1000

        return res


# ==============================================================================
# ENGINE LOADING — finds models / scalers / feature list under a base path
# ==============================================================================
def load_engine(base_path, log):
    """
    Builds an OptionPricingEngine from files found under `base_path`.
    `log` is a callable(str) used to report progress/warnings to the GUI.
    Returns (engine, available_mlp2_types, device, model_dir, data_path).
    """
    base_path = base_path.strip().rstrip("/\\")
    if not base_path:
        raise ValueError("Base path is empty. Please enter or browse to a folder.")
    if not os.path.isdir(base_path):
        raise ValueError(f"Base path does not exist or is not a folder:\n{base_path}")

    model_dir  = os.path.join(base_path, MODELS_SUBDIR)
    data_path  = os.path.join(base_path, DATA_SPLITS_SUBPATH)
    config_path = os.path.join(model_dir, CONFIG_NAME)
    scaler_cache_path = os.path.join(model_dir, SCALER_CACHE_NAME)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    # Feature columns
    feature_cols = list(DEFAULT_FEATURE_COLS)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            feature_cols = cfg.get('feature_cols', feature_cols)
            log(f"Loaded {len(feature_cols)} feature columns from config.json")
        except Exception as e:
            log(f"Could not read config.json ({e}); using built-in feature list.")
    else:
        log("config.json not found next to models; using built-in feature list.")

    # MLP2 model weights
    models = {}
    for ot in ['call', 'put']:
        weights_path = os.path.join(model_dir, f'mlp2_{ot}_final.pt')
        if os.path.exists(weights_path):
            try:
                model = MLP2(len(feature_cols)).to(device)
                state = torch.load(weights_path, map_location=device, weights_only=False)
                model.load_state_dict(state)
                model.eval()
                models[ot] = model
                log(f"Loaded MLP2 [{ot}] weights from {weights_path}")
            except Exception as e:
                log(f"Failed to load MLP2 [{ot}] weights: {e}")
        else:
            log(f"No trained weights found for [{ot}] at {weights_path} "
                f"(MLP2 pricing for {ot}s will be unavailable).")

    '''scalers
    Prefer a small cached scalers-only file (fast). Fall back to extracting
    scalers out of the full processed_data_splits.joblib (slower, since
    that file also contains the full train/val/test arrays), then cache
    them for next time so future startups are instant. '''
    scalers = {}
    if os.path.exists(scaler_cache_path):
        try:
            scalers = joblib.load(scaler_cache_path)
            log(f"Loaded cached scalers from {scaler_cache_path}")
        except Exception as e:
            log(f"Could not read cached scalers ({e}); will try full data file.")

    missing = [ot for ot in models if ot not in scalers]
    if missing and os.path.exists(data_path):
        log(f"Extracting scalers for {missing} from {data_path} (may take a moment)...")
        try:
            data_splits = joblib.load(data_path)
            for ot in missing:
                if ot in data_splits and 'scaler' in data_splits[ot]:
                    scalers[ot] = data_splits[ot]['scaler']
            try:
                joblib.dump(scalers, scaler_cache_path)
                log(f"Cached scalers to {scaler_cache_path} for faster startup next time.")
            except Exception as e:
                log(f"Could not write scaler cache ({e}); will re-extract next time.")
            del data_splits
        except Exception as e:
            log(f"Failed to load {data_path} for scalers: {e}")
    elif missing:
        log(f"No scaler cache and no data file found for {missing}; "
            f"MLP2 pricing for these option types will be skipped.")

    usable_models = {ot: m for ot, m in models.items() if ot in scalers}
    skipped = set(models) - set(usable_models)
    if skipped:
        log(f"Note: weights were found for {sorted(skipped)} but no matching "
            f"scaler — excluding from MLP2 pricing.")

    engine = OptionPricingEngine(usable_models, scalers, feature_cols, device)
    return engine, sorted(usable_models.keys()), device, model_dir, data_path


# ==============================================================================
# SMALL PARSING HELPERS FOR THE GUI
# ==============================================================================
def _to_float_or_none(s):
    s = (s or "").strip()
    return float(s) if s else None


def _to_float_or_zero(s):
    s = (s or "").strip()
    return float(s) if s else 0.0


def _to_required_float(s, field_name):
    s = (s or "").strip()
    if not s:
        raise ValueError(f"'{field_name}' is required.")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"'{field_name}' must be a number (got '{s}').")


# ==============================================================================
# GUI APPLICATION
# ==============================================================================
class OptionPricingApp(tk.Tk):
    MODEL_ORDER = [
        ("MLP2", "MLP2_ms"),
        ("BlackScholes", "BS_ms"),
        ("Binomial", "Binom_ms"),
        ("MonteCarlo", "MC_ms"),
        ("Heston", "Heston_ms"),
    ]
    CHART_COLORS = {
        "MLP2": "#2563eb", "BlackScholes": "#16a34a", "Binomial": "#d97706",
        "MonteCarlo": "#7c3aed", "Heston": "#dc2626",
    }

    def __init__(self):
        super().__init__()
        self.title("Option Pricing Engine — MLP2 / Black-Scholes / Binomial / Monte Carlo / Heston")
        self.geometry("1220x960")
        self.minsize(1080, 760)

        self.engine = None
        self.available_mlp2 = []
        self.device = None

        self._build_style()
        self._build_settings_panel()
        self._build_log_panel()

        body = ttk.Frame(self, padding=(10, 4, 10, 10))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_input_panel(body)
        self._build_results_panel(body)

        self._load_settings()

    # Style
    
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9))
        style.configure("Treeview", rowheight=24, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    # 1. settings panel
    def _build_settings_panel(self):
        frame = ttk.LabelFrame(self, text="1. Data location", padding=(10, 8))
        frame.pack(fill="x", padx=10, pady=(10, 4))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Base path:").grid(row=0, column=0, sticky="w")
        self.base_path_var = tk.StringVar(value=DEFAULT_BASE_PATH)
        ttk.Entry(frame, textvariable=self.base_path_var, width=70).grid(
            row=0, column=1, sticky="ew", padx=(6, 6))

        ttk.Button(frame, text="Browse...", command=self._browse_base_path).grid(
            row=0, column=2, padx=(0, 6))
        self.load_button = ttk.Button(frame, text="Load / Reload Engine",
                                       command=self.on_load_engine)
        self.load_button.grid(row=0, column=3)

        hint = (f"Expected under base path:  {MODELS_SUBDIR}/mlp2_call_final.pt, "
                f"{MODELS_SUBDIR}/mlp2_put_final.pt, {MODELS_SUBDIR}/{CONFIG_NAME}, "
                f"{DATA_SPLITS_SUBPATH}")
        ttk.Label(frame, text=hint, foreground="#666", font=("Segoe UI", 8)).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))

        self.status_var = tk.StringVar(value="Engine not loaded yet.")
        ttk.Label(frame, textvariable=self.status_var, style="Status.TLabel").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

    def _browse_base_path(self):
        folder = filedialog.askdirectory(title="Select base data folder")
        if folder:
            self.base_path_var.set(folder)

    # 2. input panel
    def _build_input_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="2. Option inputs", padding=(10, 8))
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        top = ttk.Frame(panel)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Option type:").grid(row=0, column=0, sticky="w")
        self.option_type_var = tk.StringVar(value="call")
        ttk.Radiobutton(top, text="Call", value="call",
                         variable=self.option_type_var).grid(row=0, column=1, padx=4)
        ttk.Radiobutton(top, text="Put", value="put",
                         variable=self.option_type_var).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="Exercise:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.exercise_var = tk.StringVar(value="european")
        ttk.Radiobutton(top, text="European", value="european",
                         variable=self.exercise_var).grid(row=1, column=1, padx=4, pady=(4, 0))
        ttk.Radiobutton(top, text="American", value="american",
                         variable=self.exercise_var).grid(row=1, column=2, padx=4, pady=(4, 0))

        notebook = ttk.Notebook(panel)
        notebook.pack(fill="both", expand=True)

        core_tab = ttk.Frame(notebook, padding=10)
        adv_tab = ttk.Frame(notebook, padding=10)
        notebook.add(core_tab, text="Core")
        notebook.add(adv_tab, text="Market data (optional)")

        self.core_vars = {}
        core_fields = [
            ("S", "Spot price (S)", "185"),
            ("K", "Strike price (K)", "190"),
            ("r", "Risk-free rate (r, e.g. 0.045)", "0.045"),
            ("sigma", "Volatility (sigma, e.g. 0.25)", "0.25"),
        ]
        row = 0
        for key, label, placeholder in core_fields:
            ttk.Label(core_tab, text=label + ":").grid(row=row, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=placeholder)
            ttk.Entry(core_tab, textvariable=var, width=18).grid(
                row=row, column=1, sticky="w", padx=(6, 0), pady=4)
            self.core_vars[key] = var
            row += 1

        ttk.Label(core_tab, text="Time to maturity T (years):").grid(
            row=row, column=0, sticky="w", pady=4)
        self.t_var = tk.StringVar(value="0.25")
        ttk.Entry(core_tab, textvariable=self.t_var, width=18).grid(
            row=row, column=1, sticky="w", padx=(6, 0), pady=4)
        row += 1

        ttk.Label(core_tab, text="  ...or days to expiry:").grid(row=row, column=0, sticky="w")
        self.days_var = tk.StringVar(value="")
        ttk.Entry(core_tab, textvariable=self.days_var, width=10).grid(
            row=row, column=1, sticky="w", padx=(6, 0))
        ttk.Button(core_tab, text="-> use as T", command=self._days_to_T).grid(
            row=row, column=2, sticky="w", padx=(6, 0))
        row += 1
        core_tab.columnconfigure(1, weight=1)

        self.adv_vars = {}
        adv_fields = [
            ("implied_vol", "Implied volatility (blank = use sigma)"),
            ("iv_atm_30d", "IV ATM 30d"),
            ("iv_atm_60d", "IV ATM 60d"),
            ("iv_90m_30d", "IV @ 90% moneyness, 30d"),
            ("iv_110m_30d", "IV @ 110% moneyness, 30d"),
            ("skew_30d", "Skew 30d (blank = auto from IV 90/110)"),
            ("skew_index", "CBOE SKEW index"),
            ("volume", "Volume"),
            ("open_interest", "Open interest"),
            ("bid_size", "Bid size"),
            ("ask_size", "Ask size"),
        ]
        for i, (key, label) in enumerate(adv_fields):
            ttk.Label(adv_tab, text=label + ":").grid(row=i, column=0, sticky="w", pady=3)
            var = tk.StringVar(value="")
            ttk.Entry(adv_tab, textvariable=var, width=16).grid(
                row=i, column=1, sticky="w", padx=(6, 0), pady=3)
            self.adv_vars[key] = var
        adv_tab.columnconfigure(1, weight=1)
        ttk.Label(adv_tab, text="Leave any field blank to fall back to the engine's defaults.",
                  foreground="#666", font=("Segoe UI", 8)).grid(
            row=len(adv_fields), column=0, columnspan=2, sticky="w", pady=(6, 0))

        btn_row = ttk.Frame(panel)
        btn_row.pack(fill="x", pady=(10, 0))
        self.price_button = ttk.Button(btn_row, text="Price Option", command=self.on_price)
        self.price_button.pack(side="left")
        ttk.Button(btn_row, text="Reset optional fields", command=self._reset_fields).pack(
            side="left", padx=(8, 0))

    def _days_to_T(self):
        try:
            days = float(self.days_var.get())
            self.t_var.set(f"{days / 365.0:.6f}")
        except ValueError:
            messagebox.showwarning("Invalid input", "Enter a numeric number of days.")

    def _reset_fields(self):
        for var in self.adv_vars.values():
            var.set("")
        self.days_var.set("")

    # 3. results panel
    def _build_results_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="3. Results", padding=(10, 8))
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)

        self.summary_var = tk.StringVar(value="No pricing run yet.")
        ttk.Label(panel, textvariable=self.summary_var, style="SubHeader.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))

        columns = ("model", "price", "time_ms")
        self.tree = ttk.Treeview(panel, columns=columns, show="headings", height=6)
        self.tree.heading("model", text="Model")
        self.tree.heading("price", text="Price ($)")
        self.tree.heading("time_ms", text="Time (ms)")
        self.tree.column("model", width=160, anchor="w")
        self.tree.column("price", width=120, anchor="e")
        self.tree.column("time_ms", width=110, anchor="e")
        self.tree.grid(row=1, column=0, sticky="ew")

        self.bid_ask_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.bid_ask_var, foreground="#444").grid(
            row=2, column=0, sticky="w", pady=(6, 6))

        self.fig, self.ax = plt.subplots(figsize=(5.6, 3.4), dpi=100)
        self.fig.subplots_adjust(bottom=0.28)
        self.canvas = FigureCanvasTkAgg(self.fig, master=panel)
        self.canvas.get_tk_widget().grid(row=3, column=0, sticky="nsew")
        panel.rowconfigure(3, weight=1)
        self._draw_empty_chart()

    def _draw_empty_chart(self):
        self.ax.clear()
        self.ax.set_title("Model price comparison")
        self.ax.set_ylabel("Price ($)")
        self.ax.text(0.5, 0.5, "Run a pricing calculation to see the chart",
                      ha="center", va="center", transform=self.ax.transAxes, color="#888")
        self.ax.set_xticks([])
        self.canvas.draw()

    # 4. log panel
    def _build_log_panel(self):
        frame = ttk.LabelFrame(self, text="Engine log", padding=(8, 4))
        frame.pack(fill="x", padx=10, pady=(0, 10))
        self.log_text = tk.Text(frame, height=6, wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill="x", side="left", expand=True)
        scrollbar = ttk.Scrollbar(frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set, state="disabled")

    def _log(self, message):
        def append():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(0, append)

    # Settings IO
    def _load_settings(self):
        if os.path.exists(APP_SETTINGS_PATH):
            try:
                with open(APP_SETTINGS_PATH, "r") as f:
                    settings = json.load(f)
                last_path = settings.get("base_path", "")
                if last_path:
                    self.base_path_var.set(last_path)
            except Exception:
                pass

    def _save_settings(self):
        try:
            with open(APP_SETTINGS_PATH, "w") as f:
                json.dump({"base_path": self.base_path_var.get().strip()}, f)
        except Exception:
            pass

    # Engine loading
    def on_load_engine(self):
        base_path = self.base_path_var.get().strip()
        self.load_button.configure(state="disabled")
        self.status_var.set("Loading engine...")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
        threading.Thread(target=self._load_engine_worker, args=(base_path,), daemon=True).start()

    def _load_engine_worker(self, base_path):
        try:
            engine, available, device, model_dir, data_path = load_engine(base_path, self._log)

            def done():
                self.engine = engine
                self.available_mlp2 = available
                self.device = device
                if available:
                    self.status_var.set(
                        f"Engine ready on {device}. MLP2 available for: {', '.join(available)}.")
                else:
                    self.status_var.set(
                        f"Engine ready on {device}. MLP2 unavailable — classical models only.")
                self.load_button.configure(state="normal")
                self._save_settings()
            self.after(0, done)
        except Exception as e:
            err = str(e)

            def fail():
                self.status_var.set(f"Failed to load engine: {err}")
                self.load_button.configure(state="normal")
                messagebox.showerror("Engine load failed", err)
            self.after(0, fail)

    # Pricing
    def on_price(self):
        if self.engine is None:
            messagebox.showwarning("Engine not loaded", "Click 'Load / Reload Engine' first.")
            return
        try:
            kwargs = self._collect_inputs()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.price_button.configure(state="disabled")
        self.summary_var.set("Pricing...")
        threading.Thread(target=self._price_worker, args=(kwargs,), daemon=True).start()

    def _collect_inputs(self):
        S = _to_required_float(self.core_vars["S"].get(), "Spot price (S)")
        K = _to_required_float(self.core_vars["K"].get(), "Strike price (K)")
        T = _to_required_float(self.t_var.get(), "Time to maturity (T)")
        r = _to_required_float(self.core_vars["r"].get(), "Risk-free rate (r)")
        sigma = _to_required_float(self.core_vars["sigma"].get(), "Volatility (sigma)")
        if S <= 0 or K <= 0:
            raise ValueError("Spot price and strike price must be positive.")
        if T <= 0:
            raise ValueError("Time to maturity must be positive.")
        if sigma <= 0:
            raise ValueError("Volatility must be positive.")

        return dict(
            S=S, K=K, T=T, r=r, sigma=sigma,
            option_type=self.option_type_var.get(),
            exercise=self.exercise_var.get(),
            implied_vol=_to_float_or_none(self.adv_vars["implied_vol"].get()),
            iv_atm_30d=_to_float_or_none(self.adv_vars["iv_atm_30d"].get()),
            iv_atm_60d=_to_float_or_none(self.adv_vars["iv_atm_60d"].get()),
            iv_90m_30d=_to_float_or_none(self.adv_vars["iv_90m_30d"].get()),
            iv_110m_30d=_to_float_or_none(self.adv_vars["iv_110m_30d"].get()),
            skew_30d=_to_float_or_none(self.adv_vars["skew_30d"].get()),
            skew_index=_to_float_or_none(self.adv_vars["skew_index"].get()),
            volume=_to_float_or_zero(self.adv_vars["volume"].get()),
            open_interest=_to_float_or_zero(self.adv_vars["open_interest"].get()),
            bid_size=_to_float_or_zero(self.adv_vars["bid_size"].get()),
            ask_size=_to_float_or_zero(self.adv_vars["ask_size"].get()),
        )

    def _price_worker(self, kwargs):
        try:
            res = self.engine.price(**kwargs)
            self.after(0, lambda: self._display_results(res))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: messagebox.showerror("Pricing failed", err))
        finally:
            self.after(0, lambda: self.price_button.configure(state="normal"))

    def _display_results(self, res):
        inp = res["inputs"]
        self.summary_var.set(
            f"S=${inp['S']:.2f}  K=${inp['K']:.2f}  T={inp['T']:.3f}y  "
            f"r={inp['r'] * 100:.2f}%  sigma={inp['sigma'] * 100:.1f}%   "
            f"-> {inp['option_type'].upper()} {inp['exercise'].capitalize()}"
        )

        for row in self.tree.get_children():
            self.tree.delete(row)

        for name, tkey in self.MODEL_ORDER:
            if name in res:
                self.tree.insert("", "end",
                                  values=(name, f"{res[name]:.4f}", f"{res.get(tkey, 0):.2f}"))

        if "MLP2_bid" in res:
            self.bid_ask_var.set(
                f"MLP2 bid/ask spread:  ${res['MLP2_bid']:.4f}  /  ${res['MLP2_ask']:.4f}")
        else:
            self.bid_ask_var.set(
                "MLP2 not available for this option type (no trained weights/scaler found).")

        self._update_chart(res)

    def _update_chart(self, res):
        names, values, colors = [], [], []
        for name, _ in self.MODEL_ORDER:
            if name in res:
                names.append(name)
                values.append(res[name])
                colors.append(self.CHART_COLORS.get(name, "#666"))

        self.ax.clear()
        bars = self.ax.bar(names, values, color=colors)
        self.ax.set_title("Model price comparison")
        self.ax.set_ylabel("Price ($)")
        self.ax.set_xticks(range(len(names)))
        self.ax.set_xticklabels(names, rotation=20, ha="right")
        for bar, val in zip(bars, values):
            self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    app = OptionPricingApp()
    app.mainloop()