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
    5. Drop redundant per-city weather (keep Calgary/Lethbridge for Chinook)
    6. Engineer lag features       (price, AIL, temperature, wind)
    7. Engineer weather flags      (8 binary flags)
    8. Define target variable      (ΔPrice = hourly absolute change)
    9. Add cyclical time features  (sin/cos for hour, day-of-week, month)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "Data"

ENERGY_PATH  = DATA_DIR / "Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv"
WEATHER_PATH = DATA_DIR / "alberta_weather_2020_2025.csv"
OUTPUT_PATH  = DATA_DIR / "energy_weather_featured.csv"


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
    Drop per-city columns that are redundant with province averages.
    Keep Calgary & Lethbridge T2M, WS50M, RH2M for Chinook detection.
    Drop all Edmonton per-city (captured by AVG).
    Drop all WS10M (redundant with WS50M, which is hub-height).
    Drop per-city solar (captured by GHI_AVG).
    """
    print("\nDropping redundant weather columns...")

    drop_cols = [
        # Edmonton per-city — redundant with provincial averages
        "T2M_EDMONTON",
        "WS10M_EDMONTON",
        "WS50M_EDMONTON",
        "RH2M_EDMONTON",
        "ALLSKY_SFC_SW_DWN_EDMONTON",
        # Imports per-city — redundant with provincial imports, and not directly relevant to SMP price
        "IMPORT_BC",
        "IMPORT_MT",
        "IMPORT_SK",
        # WS10M everywhere — redundant with WS50M (hub-height)
        "WS10M_CALGARY",
        "WS10M_LETHBRIDGE",
        "WS10M_AVG",
        # Per-city solar — captured by GHI_AVG
        "ALLSKY_SFC_SW_DWN_CALGARY",
        "ALLSKY_SFC_SW_DWN_LETHBRIDGE",
        # Per-city RH — Chinook no longer uses RH (NASA POWER too coarse),
        # and RH2M_AVG captures the general humidity signal
        "RH2M_CALGARY",
        "RH2M_LETHBRIDGE",
        # Price Forecast — not a weather column, and we want to predict actual price without peeking at forecasts
        "HOUR_AHEAD_POOL_PRICE_FORECAST"
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
    Add lag features for LSTM forecasting.

    Price lags:  t-1 (autocorrelation), t-24 (daily), t-168 (weekly)
    AIL lags:    t-1 (demand momentum), t-24 (demand norm comparison)
    Weather lags: T2M_AVG t-24 (heating/cooling ramp), WS50M_AVG t-1 (wind ramp)
    """
    print("\nAdding lag features...")

    lag_specs = [
        ("ACTUAL_POOL_PRICE", 1,   "price_lag_1"),
        ("ACTUAL_POOL_PRICE", 24,  "price_lag_24"),
        ("ACTUAL_POOL_PRICE", 168, "price_lag_168"),
        ("ACTUAL_AIL",        1,   "AIL_lag_1"),
        ("ACTUAL_AIL",        24,  "AIL_lag_24"),
        ("T2M_AVG",           24,  "T2M_AVG_lag_24"),
        ("WS50M_AVG",         1,   "WS50M_AVG_lag_1"),
    ]

    for source_col, lag, new_name in lag_specs:
        df[new_name] = df[source_col].shift(lag)
        print(f"  {new_name:<25s} = {source_col} shifted by {lag}h")

    # Drop rows where the largest lag (168) creates NaNs
    n_before = len(df)
    df = df.dropna(subset=[name for _, _, name in lag_specs]).reset_index(drop=True)
    n_after = len(df)
    print(f"  Rows dropped (lag warm-up): {n_before - n_after}")
    print(f"  Shape after lags: {df.shape}")

    return df


# =============================================================================
# STAGE 7 — Weather flags
# =============================================================================
def add_weather_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary weather flags with clear causal links to SMP.

    Temperature flags:
        is_heating_season       T2M_AVG < 10°C       — elevated heating demand
        is_cooling_season       T2M_AVG > 20°C        — AC load increase
        is_cold_snap            T2M_AVG < -20°C       — demand surge + supply risk
        is_temp_dropping_fast   ΔT24h < -8°C          — demand ramp surprise

    Wind flags:
        is_low_wind             WS50M_AVG < 4 m/s     — below turbine cut-in
        is_high_wind            WS50M_AVG > 12 m/s    — heavy wind generation

    Solar flag:
        is_solar_generating     GHI_AVG > 50 W/m²     — daytime solar output

    Chinook flag (Calgary-specific):
        is_chinook              T2M_CALGARY jumps >10°C in 24h
                                + WS50M_CALGARY > 10 m/s
                                + RH2M_CALGARY < 40%
    """
    print("\nAdding weather flags...")

    # ── Temperature flags ────────────────────────────────────────────────
    df["is_heating_season"] = (df["T2M_AVG"] < 10).astype(int)
    df["is_cooling_season"] = (df["T2M_AVG"] > 20).astype(int)
    df["is_cold_snap"]      = (df["T2M_AVG"] < -20).astype(int)

    # Rapid cooling: current temp vs. 24h ago (T2M_AVG_lag_24 already exists)
    df["is_temp_dropping_fast"] = (
        (df["T2M_AVG"] - df["T2M_AVG_lag_24"]) < -8
    ).astype(int)

    # ── Wind flags ───────────────────────────────────────────────────────
    df["is_low_wind"]  = (df["WS50M_AVG"] < 4).astype(int)
    df["is_high_wind"] = (df["WS50M_AVG"] > 12).astype(int)

    # ── Solar flag ───────────────────────────────────────────────────────
    df["is_solar_generating"] = (df["GHI_AVG"] > 50).astype(int)

    # ── Chinook flag (requires Calgary per-city columns) ─────────────────
    # Chinook = rapid warming in Calgary + strong wind, restricted to winter.
    # NASA POWER's coarse grid doesn't capture the RH drop well, so we use
    # temp rise + wind + winter season as the detection criteria.
    T2M_calgary_lag_24 = df["T2M_CALGARY"].shift(24)
    is_winter_month = df["Date_Begin_Local"].dt.month.isin([10, 11, 12, 1, 2, 3])
    df["is_chinook"] = (
        ((df["T2M_CALGARY"] - T2M_calgary_lag_24) > 10)
        & (df["WS50M_CALGARY"] > 8)
        & is_winter_month
    ).astype(int).fillna(0).astype(int)

    # ── Report sample counts ─────────────────────────────────────────────
    flag_cols = [c for c in df.columns if c.startswith("is_")]
    print(f"\n  {'Flag':<30s} {'True':>8s} {'%':>8s}")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 8}")
    for col in flag_cols:
        n_true = df[col].sum()
        pct = 100 * n_true / len(df)
        print(f"  {col:<30s} {n_true:>8,.0f} {pct:>7.1f}%")

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
