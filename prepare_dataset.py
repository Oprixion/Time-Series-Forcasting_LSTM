"""
prepare_dataset.py — Join, Clean & Engineer Alberta Energy + Weather Data
=========================================================================
Joins the AESO hourly energy data with NASA POWER weather data, filters to
province-level columns, engineers lag features and weather flags, and
outputs a clean dataset ready for LSTM forecasting of SMP pool price.

Pipeline stages:
    1. Load & filter energy data  (drop plant-specific + export columns)
    2. Load & fix weather data    (DST spring-forward remap)
    3. Join on local timestamp
    4. Clean NaNs & dedup DST fall-back rows
    5. Drop redundant weather columns (all per-city; retain T2M_AVG + WS50M_AVG for Stage 7)
    6. Engineer lag features       (price lags t-1/24/168 + 24h rolling volatility)
    7. Engineer model features     (wind proxy, cold snap, T² term, AIL×T interaction;
                                    drop WS50M_AVG after use)
    8. Define target variable      (ΔPrice = hourly absolute change)
    9. Add cyclical time features  (sin/cos for hour, day-of-week, month)
   10. Chronological train/test split (80% train, 20% test) → train.csv, test.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "Data"

ENERGY_PATH  = DATA_DIR / "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv"
WEATHER_PATH = DATA_DIR / "alberta_weather_2020_2025.csv"
OUTPUT_PATH  = DATA_DIR / "energy_weather_featured.csv"
TRAIN_PATH   = DATA_DIR / "train.csv"
TEST_PATH    = DATA_DIR / "test.csv"

TRAIN_RATIO  = 0.80   # chronological split — no shuffling


# =============================================================================
# STAGE 1 — Load energy data
# =============================================================================
def load_energy(path: Path) -> pd.DataFrame:
    """Load AESO energy data and keep only province-level columns."""
    print(f"Loading energy data from {path.name}...")
    df = pd.read_csv(path, parse_dates=["Date_Begin_Local"])

    keep_cols = [
        "Date_Begin_Local",
        "ACTUAL_POOL_PRICE",
        "ACTUAL_AIL",
        "HOUR_AHEAD_POOL_PRICE_FORECAST",
        "IMPORT_BC",
        "IMPORT_MT",
        "IMPORT_SK",
    ]

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in energy data: {missing}")

    df = df[keep_cols]
    print(f"  Energy shape after column filter: {df.shape}")
    print(f"  Date range: {df['Date_Begin_Local'].min()} → {df['Date_Begin_Local'].max()}")
    return df


# =============================================================================
# STAGE 2 — Load weather data
# =============================================================================
def load_weather(path: Path) -> pd.DataFrame:
    """Load NASA POWER weather data and fix DST spring-forward timestamps."""
    print(f"\nLoading weather data from {path.name}...")
    df = pd.read_csv(path, parse_dates=["datetime_mst"])

    print(f"  Weather shape: {df.shape}")
    print(f"  Date range: {df['datetime_mst'].min()} → {df['datetime_mst'].max()}")

    # NASA POWER uses true MST (no DST), so it has 02:00 rows on spring-forward
    # Sundays. AESO clock-time skips 02:00 → 03:00. Remap to align.
    dst_spring_forward = pd.to_datetime([
        "2020-03-08 02:00", "2021-03-14 02:00", "2022-03-13 02:00",
        "2023-03-12 02:00", "2024-03-10 02:00", "2025-03-09 02:00",
    ])

    mask = df["datetime_mst"].isin(dst_spring_forward)
    n_fixed = mask.sum()
    df.loc[mask, "datetime_mst"] += pd.Timedelta(hours=1)
    print(f"  DST spring-forward rows remapped: {n_fixed}")

    return df


# =============================================================================
# STAGE 3 — Join datasets
# =============================================================================
def join_datasets(energy: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Left-join energy onto weather using local timestamps."""
    print("\nJoining datasets...")
    df = energy.merge(
        weather,
        left_on="Date_Begin_Local",
        right_on="datetime_mst",
        how="left",
    ).drop(columns="datetime_mst")

    n_missing = df["T2M_AVG"].isna().sum()
    print(f"  Joined shape: {df.shape}")
    print(f"  Rows with missing weather data: {n_missing}")

    if n_missing > 0:
        missing_times = df.loc[df["T2M_AVG"].isna(), "Date_Begin_Local"].tolist()
        print(f"  Missing timestamps: {missing_times}")

    return df


# =============================================================================
# STAGE 4 — Clean & dedup
# =============================================================================
def clean_and_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Fill weather NaNs and average DST fall-back duplicate timestamps."""
    print("\nCleaning dataset...")

    # Forward-fill then back-fill weather edge-case NaNs
    weather_cols = [c for c in df.columns if c not in [
        "Date_Begin_Local", "ACTUAL_POOL_PRICE", "ACTUAL_AIL",
        "HOUR_AHEAD_POOL_PRICE_FORECAST", "IMPORT_BC", "IMPORT_MT", "IMPORT_SK"
    ]]

    n_before = df[weather_cols].isna().sum().sum()
    df[weather_cols] = df[weather_cols].ffill()
    df[weather_cols] = df[weather_cols].bfill()
    n_after = df[weather_cols].isna().sum().sum()
    print(f"  Weather NaNs filled: {n_before} → {n_after}")

    # DST fall-back: AESO records two rows for same clock hour in November.
    # Average them to keep a strictly monotonic time index for LSTM.
    n_dupes = df["Date_Begin_Local"].duplicated().sum()
    if n_dupes > 0:
        print(f"  DST fall-back duplicate timestamps found: {n_dupes}")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df = df.groupby("Date_Begin_Local", as_index=False)[numeric_cols].mean()
        print(f"  Shape after averaging duplicates: {df.shape}")
        assert df["Date_Begin_Local"].is_monotonic_increasing, \
            "Timestamps are not monotonically increasing after dedup!"
        print("  Timestamps are monotonically increasing.")

    total_nans = df.isna().sum()
    if total_nans.any():
        print(f"  Remaining NaNs:\n{total_nans[total_nans > 0]}")
    else:
        print("  No remaining NaN values.")

    return df


# =============================================================================
# STAGE 5 — Drop redundant weather columns
# =============================================================================
def drop_redundant_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all per-city and redundant weather columns.

    Retained for downstream stages:
        T2M_AVG     — dominant temperature signal (LASSO coeff -78.11); feeds T2M_sq and AIL_T2M
        WS50M_AVG   — hub-height wind speed; feeds wind_power_proxy and is_low_wind (dropped after Stage 7)

    Everything else is either captured by those two averages, feeds no remaining engineered feature,
    or was shown to contribute negligible signal (RH2M_AVG, GHI_AVG).
    """
    print("\nDropping redundant weather columns...")

    drop_cols = [
        # Edmonton per-city — redundant with provincial averages
        "T2M_EDMONTON",
        "WS10M_EDMONTON",
        "WS50M_EDMONTON",
        "RH2M_EDMONTON",
        "ALLSKY_SFC_SW_DWN_EDMONTON",
        # Inter-provincial imports — not directly relevant to SMP price formation
        "IMPORT_BC",
        "IMPORT_MT",
        "IMPORT_SK",
        # WS10M everywhere — redundant with WS50M (hub-height)
        "WS10M_CALGARY",
        "WS10M_LETHBRIDGE",
        "WS10M_AVG",
        # Per-city solar — GHI_AVG is also dropped; solar signal captured by hour/month encodings
        "ALLSKY_SFC_SW_DWN_CALGARY",
        "ALLSKY_SFC_SW_DWN_LETHBRIDGE",
        # GHI_AVG — solar irradiance dropped; intra-day and seasonal cycles already encoded
        # by hour_sin/cos and month_sin/cos, which are more compact and non-redundant
        "GHI_AVG",
        # Per-city RH and provincial RH — weakest weather feature (LASSO coeff -0.88);
        # temperature and wind fully dominate the weather signal
        "RH2M_CALGARY",
        "RH2M_LETHBRIDGE",
        "RH2M_AVG",
        # Per-city Calgary & Lethbridge T2M and WS50M — previously retained for Chinook
        # detection only. Chinook flag dropped (0.4% prevalence, ablation-negative,
        # imperfect detection). No remaining use for per-city columns.
        "T2M_CALGARY",
        "WS50M_CALGARY",
        "T2M_LETHBRIDGE",
        "WS50M_LETHBRIDGE",
        # Price Forecast — excluded to prevent peeking at AESO's own forward estimate
        "HOUR_AHEAD_POOL_PRICE_FORECAST",
    ]

    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)
    print(f"  Dropped {len(existing)} columns: {existing}")
    print(f"  Shape after drop: {df.shape}")

    return df


# =============================================================================
# STAGE 6 — Lag features
# =============================================================================
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price lag features and rolling volatility.

    Price lags:
        t-1   — strongest single predictor (LASSO coeff -38.54); anchors the delta_price target.
        t-24  — same-hour-yesterday daily cycle (LASSO coeff +14.40).
        t-168 — same-hour-last-week weekly cycle (LASSO coeff +5.94); confirmed independent
                signal even after controlling for dow_sin/cos encodings.

    Rolling volatility:
        price_rolling_std_24 — 24h std of price_lag_1. Encodes whether the past day was
        quiet or turbulent. Volatility clusters in energy markets; a turbulent recent window
        predicts higher spike likelihood. Computed on lagged prices only — no leakage.

    AIL lags and weather lags (AIL_lag_1, AIL_lag_24, T2M_AVG_lag_24, WS50M_AVG_lag_1)
    are excluded: the LSTM's 48h lookback window provides the same information through
    the sequential input, making explicit lags of these variables redundant.
    """
    print("\nAdding lag features...")

    lag_specs = [
        ("ACTUAL_POOL_PRICE", 1,   "price_lag_1"),
        ("ACTUAL_POOL_PRICE", 24,  "price_lag_24"),
        ("ACTUAL_POOL_PRICE", 168, "price_lag_168"),
    ]

    for source_col, lag, new_name in lag_specs:
        df[new_name] = df[source_col].shift(lag)
        print(f"  {new_name:<25s} = {source_col} shifted by {lag}h")

    # Rolling 24h price volatility — computed on price_lag_1 (fully historical,
    # no leakage). A 24-window captures yesterday's market turbulence, which is
    # the most relevant context for whether the next hour is likely to spike.
    # Window=24 is preferable to 48 because volatility is a short-term signal —
    # older history is already encoded in the lag features and LSTM sequence.
    df["price_rolling_std_24"] = df["price_lag_1"].rolling(window=24).std()
    print(f"  {'price_rolling_std_24':<25s} = rolling(24).std() of price_lag_1")

    # Drop rows where any lag or rolling feature creates NaNs.
    # The largest lag (168h) dominates — rolling(24) NaNs are already covered.
    n_before = len(df)
    drop_subset = [name for _, _, name in lag_specs] + ["price_rolling_std_24"]
    df = df.dropna(subset=drop_subset).reset_index(drop=True)
    n_after = len(df)
    print(f"  Rows dropped (lag warm-up): {n_before - n_after}")
    print(f"  Shape after lags: {df.shape}")

    return df


# =============================================================================
# STAGE 7 — Weather flags
# =============================================================================
def add_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the final model features from raw weather and demand columns.

    Wind (supply side):
        is_low_wind        WS50M_AVG < 4 m/s   — binary cut-in flag. The regime change at
                                                   cut-in speed is a step function that the
                                                   continuous proxy cannot represent alone.
        wind_power_proxy   WS50M_AVG³           — continuous supply signal (P ∝ v³, Betz law).
                                                   Alberta wind generators bid at $0/MWh;
                                                   higher wind output directly suppresses SMP.
                                                   The cube transformation correctly weights
                                                   high-speed events over low-speed ones.

    Temperature (demand + supply side):
        is_cold_snap       T2M_AVG < -20°C      — extreme cold simultaneously surges heating
                                                   demand AND risks gas freeze-offs (supply risk).
                                                   Ablation confirmed signal; the only binary flag
                                                   encoding an unobserved supply-side variable.
        T2M_sq             T2M_AVG²             — temperature has a U-shaped relationship with
                                                   price: both extreme cold and extreme heat drive
                                                   prices up. The squared term captures this
                                                   curvature (LASSO coeff +10.22 on standardised
                                                   features).

    Demand × temperature interaction:
        AIL_T2M            ACTUAL_AIL × T2M_AVG — second-strongest LASSO feature (coeff +74.22).
                                                   Cold temperatures only drive prices up because
                                                   demand is simultaneously high; the interaction
                                                   encodes this joint effect that neither feature
                                                   alone can express.

    WS50M_AVG is dropped at the end of this stage — its information is fully absorbed
    into wind_power_proxy and is_low_wind.
    """
    print("\nEngineering model features...")

    # ── Wind features ────────────────────────────────────────────────────
    df["is_low_wind"]      = (df["WS50M_AVG"] < 4).astype(int)
    df["wind_power_proxy"] = df["WS50M_AVG"] ** 3

    # ── Temperature features ─────────────────────────────────────────────
    df["is_cold_snap"] = (df["T2M_AVG"] < -20).astype(int)
    df["T2M_sq"]       = df["T2M_AVG"] ** 2

    # ── Demand × temperature interaction ─────────────────────────────────
    df["AIL_T2M"] = df["ACTUAL_AIL"] * df["T2M_AVG"]

    # ── Drop WS50M_AVG — absorbed into wind features ─────────────────────
    df = df.drop(columns=["WS50M_AVG"])

    # ── Report ────────────────────────────────────────────────────────────
    print(f"  is_low_wind      : {df['is_low_wind'].sum():>6,} samples  ({100*df['is_low_wind'].mean():.1f}%)")
    print(f"  is_cold_snap     : {df['is_cold_snap'].sum():>6,} samples  ({100*df['is_cold_snap'].mean():.1f}%)")
    print(f"  wind_power_proxy : mean={df['wind_power_proxy'].mean():.1f},  max={df['wind_power_proxy'].max():.1f}")
    print(f"  T2M_sq           : mean={df['T2M_sq'].mean():.1f}")
    print(f"  AIL_T2M          : mean={df['AIL_T2M'].mean():.1f}")

    return df


# =============================================================================
# STAGE 8 — Target variable
# =============================================================================
def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define LSTM target: absolute hourly change in SMP pool price.
    delta_price = ACTUAL_POOL_PRICE(t) - ACTUAL_POOL_PRICE(t-1)
    price_lag_1 already exists, so this is just the difference.
    """
    print("\nAdding target variable (delta_price)...")

    df["delta_price"] = df["ACTUAL_POOL_PRICE"] - df["price_lag_1"]

    print(f"  delta_price stats:")
    print(f"    mean:  {df['delta_price'].mean():>+10.2f}")
    print(f"    std:   {df['delta_price'].std():>10.2f}")
    print(f"    min:   {df['delta_price'].min():>+10.2f}")
    print(f"    max:   {df['delta_price'].max():>+10.2f}")
    print(f"    nulls: {df['delta_price'].isna().sum()}")

    return df


# =============================================================================
# STAGE 9 — Cyclical time features
# =============================================================================
def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour-of-day, day-of-week, and month as sin/cos pairs extracted
    from Date_Begin_Local.

    Raw integer encoding (e.g. hour 0–23) creates an artificial discontinuity
    where hour 23 and hour 0 appear numerically distant despite being 1 hour
    apart. Sin/cos encoding wraps the cycle so that the distance between any
    two time values reflects their true temporal proximity.

    Periods:
        hour_of_day  → period 24
        day_of_week  → period 7  (0 = Monday)
        month        → period 12

    Date_Begin_Local is retained in the dataframe as a reference index but
    should be excluded from the LSTM input feature array.
    """
    print("\nAdding cyclical time features...")

    dt = df["Date_Begin_Local"].dt

    # Hour of day (period = 24)
    hour = dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week (period = 7, Monday = 0)
    dow = dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Month (period = 12)
    month = dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    time_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    print(f"  Added: {time_features}")
    print(f"  Shape after time features: {df.shape}")
    print(f"  Note: Date_Begin_Local retained as reference index only — exclude from LSTM inputs.")

    return df


# =============================================================================
# STAGE 10 — Chronological train / test split
# =============================================================================
def split_and_save(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO) -> None:
    """
    Split the fully featured dataset into train and test sets chronologically
    and write each to its own CSV file.

    The split is strictly chronological — no shuffling — to preserve the
    time-series structure and prevent any future data leaking into training.
    The test set is held out entirely for final model comparison (LSTM vs ARIMA)
    and must not be used for hyperparameter tuning or early stopping decisions.

    Split:
        train.csv — first 80% of rows by timestamp
        test.csv  — remaining 20% of rows by timestamp
    """
    print("\nSplitting into train / test sets...")

    split_idx  = int(len(df) * train_ratio)
    train      = df.iloc[:split_idx].reset_index(drop=True)
    test       = df.iloc[split_idx:].reset_index(drop=True)

    train_start = train["Date_Begin_Local"].iloc[0]
    train_end   = train["Date_Begin_Local"].iloc[-1]
    test_start  = test["Date_Begin_Local"].iloc[0]
    test_end    = test["Date_Begin_Local"].iloc[-1]

    print(f"  Train : {len(train):>6,} rows  |  {train_start} → {train_end}")
    print(f"  Test  : {len(test):>6,} rows  |  {test_start}  → {test_end}")
    print(f"  Split ratio: {len(train)/len(df)*100:.1f}% / {len(test)/len(df)*100:.1f}%")

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH,   index=False)

    print(f"  Saved train → {TRAIN_PATH}  ({TRAIN_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  Saved test  → {TEST_PATH}   ({TEST_PATH.stat().st_size / 1e6:.1f} MB)")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("Alberta Energy + Weather — Full Feature Engineering Pipeline")
    print("=" * 70)

    # Stages 1-2: Load
    energy = load_energy(ENERGY_PATH)
    weather = load_weather(WEATHER_PATH)

    # Stage 3: Join
    df = join_datasets(energy, weather)

    # Stage 4: Clean & dedup
    df = clean_and_dedup(df)

    # Stage 5: Drop redundant weather
    df = drop_redundant_weather(df)

    # Stage 6: Lag features
    df = add_lag_features(df)

    # Stage 7: Weather flags
    df = add_weather_flags(df)

    # Stage 8: Target variable
    df = add_target(df)

    # Stage 9: Cyclical time features
    df = add_cyclical_time_features(df)

    # Stage 10: Train / test split
    split_and_save(df)

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Date_Begin_Local'].min()} → {df['Date_Begin_Local'].max()}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        nulls = df[col].isna().sum()
        print(f"  {i:2d}. {col:<35s} {str(dtype):<12s} nulls={nulls}")

    print(f"\nTarget (delta_price) stats:")
    print(df["delta_price"].describe())

    # ── Save ─────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"File size: {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
