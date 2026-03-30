# Data Dictionary — Alberta SMP Price Forecasting

---

## Raw Data Sources

### 1. AESO Hourly Generation, Pool Price & AIL
**File:** `Hourly_Metered_Volumes_and_Pool_Price_and_AIL_2020-Jul2025.csv`
**Source:** [Alberta Electric System Operator (AESO)](https://www.aeso.ca/market/market-and-system-reporting/data-requests/hourly-generation-metered-volumes-and-pool-price-and-ail-data-2001-to-july-2025/)
**Coverage:** January 2020 — July 2025, hourly resolution
**How to obtain:** Navigate to the AESO link above and download the CSV for the desired date range. No API; manual download from the AESO data portal.

This file contains hourly metered generation volumes for ~225 individual power plants across Alberta, the hourly System Marginal Price (SMP / pool price), Alberta Internal Load (AIL), AESO's hour-ahead price forecast, and inter-provincial import/export volumes with BC, Montana, and Saskatchewan. Timestamps are in local Alberta clock time (MST/MDT).

---

### 2. Alberta Weather Data
**File:** `alberta_weather_2020_2025.csv`
**Source:** [NASA POWER (Prediction Of Worldwide Energy Resources)](https://power.larc.nasa.gov/)
**Coverage:** January 2020 — July 2025, hourly resolution
**How to obtain:** The fetch script was accidentally deleted. Data can be re-pulled via the NASA POWER API at `https://power.larc.nasa.gov/api/temporal/hourly/point` with the parameters below for each station, then joined into a single file. NASA POWER uses true MST (UTC−7, no daylight saving), which creates a 1-hour offset against AESO's clock-time timestamps on DST spring-forward days — see `prepare_dataset.py` for the remap logic.

**Stations and coordinates used:**

| Station | Latitude | Longitude |
|---|---|---|
| Calgary | 51.05 | -114.07 |
| Edmonton | 53.55 | -113.49 |
| Lethbridge | 49.70 | -112.83 |

**API parameters:** `T2M`, `WS10M`, `WS50M`, `RH2M`, `ALLSKY_SFC_SW_DWN`
**Community:** `RE` (Renewable Energy)
**Temporal:** `hourly`

Weather variables included:
- `T2M` — Air temperature at 2m height (°C)
- `WS10M` — Wind speed at 10m height (m/s)
- `WS50M` — Wind speed at 50m height (m/s), closer to turbine hub height
- `RH2M` — Relative humidity at 2m (%)
- `ALLSKY_SFC_SW_DWN` — Downwelling shortwave solar irradiance (W/m²)

Provided per station (Calgary, Edmonton, Lethbridge) plus provincial averages (`T2M_AVG`, `WS50M_AVG`, `RH2M_AVG`, `GHI_AVG`).

---

## Processed Dataset

**File:** `energy_weather_featured.csv`
**Script:** `prepare_dataset.py`
**Coverage:** January 8, 2020 — July 31, 2025 (first 168 rows dropped for lag warm-up)
**Shape:** 48,762 rows × 20 columns

The pipeline also outputs two model-ready split files:

| File | Rows | Period | Purpose |
|---|---|---|---|
| `train.csv` | ~39,010 (80%) | 2020-01-08 → ~2024-03 | Model training and internal validation |
| `test.csv` | ~9,752 (20%) | ~2024-03 → 2025-07-31 | Final held-out evaluation only |

The split is strictly chronological (no shuffling). `test.csv` must not be used for hyperparameter tuning or early stopping — it is reserved for the final LSTM vs ARIMA comparison.

---

### Base Features

| Column | Type | Description |
|---|---|---|
| `Date_Begin_Local` | datetime | Hour-beginning timestamp in local Alberta clock time. Reference index only — exclude from LSTM inputs. |
| `ACTUAL_POOL_PRICE` | float | System Marginal Price (SMP) in $/MWh. The raw target variable. |
| `ACTUAL_AIL` | float | Alberta Internal Load in MW. Total provincial electricity demand. |

---

### Weather & Demand Features

All per-city columns (Calgary, Edmonton, Lethbridge) are dropped in the pipeline. `WS50M_AVG` is used internally to compute `wind_power_proxy` and `is_low_wind`, then dropped before output.

| Column | Type | Description |
|---|---|---|
| `T2M_AVG` | float | Province-average 2m air temperature (°C). Dominant LASSO feature (coeff −78.11 on standardised features). Drives heating and cooling demand. |
| `T2M_sq` | float | `T2M_AVG²` — captures the U-shaped temperature–price relationship. Both extreme cold (heating demand) and extreme heat (cooling demand) drive prices up; mild temperatures are cheapest. Linear temperature alone cannot encode this curvature. LASSO coeff +10.22. |
| `AIL_T2M` | float | `ACTUAL_AIL × T2M_AVG` — demand–temperature interaction. Second-strongest LASSO feature (coeff +74.22). Cold temperatures only raise prices when load is simultaneously high; this term encodes the joint effect that neither AIL nor T2M_AVG can express independently. |
| `wind_power_proxy` | float | `WS50M_AVG³` — continuous wind supply signal. Wind turbine power output scales with the cube of wind speed (P ∝ v³, Betz law). Alberta wind generators bid at $0/MWh, so higher wind output directly suppresses the clearing price. The cube transformation correctly weights high-speed events over low-speed ones. |

---

### Lag Features

AIL lags and weather lags are excluded — the LSTM's 48h lookback window already provides this information through the sequential input, making explicit lags of these variables redundant.

| Column | Type | Description |
|---|---|---|
| `price_lag_1` | float | SMP pool price 1 hour prior (t−1). Strongest single predictor (LASSO coeff −38.54). Captures short-term autocorrelation and anchors the `delta_price` target definition. |
| `price_lag_24` | float | SMP pool price 24 hours prior (t−24). Encodes same-hour-yesterday daily seasonality (LASSO coeff +14.40). Lets the model compare whether the current price is high or low relative to this time yesterday. |
| `price_lag_168` | float | SMP pool price 168 hours prior (t−168). Encodes same-hour-last-week weekly seasonality — weekday vs. weekend load patterns (LASSO coeff +5.94, confirmed independent of `dow_sin/cos`). |
| `price_rolling_std_24` | float | Rolling 24-hour standard deviation of `price_lag_1`. Encodes recent market volatility — whether the past day was quiet or turbulent — as a predictor of spike likelihood. Computed on lagged prices only; no data leakage. |

---

### Binary Flags

Binary (0/1) features retained only where they encode a qualitative regime change that the continuous features cannot represent.

| Column | True Samples | % | Condition | Rationale |
|---|---|---|---|---|
| `is_low_wind` | ~10,077 | ~20.7% | `WS50M_AVG < 4 m/s` | Below turbine cut-in speed, Alberta's wind fleet drops to near-zero output removing low-cost supply. The power–speed relationship is a near-step-function at cut-in; `wind_power_proxy` (continuous) does not capture this discontinuity alone. |
| `is_cold_snap` | ~1,257 | ~2.6% | `T2M_AVG < -20°C` | The only binary flag retained because it encodes an *unobserved* supply-side variable: extreme cold simultaneously surges heating demand AND risks natural gas freeze-offs, knocking generation offline. No continuous feature in the set proxies the supply-risk component. Ablation confirmed positive signal. |

---

### Target Variable

| Column | Type | Description |
|---|---|---|
| `delta_price` | float | Absolute hourly change in SMP: `ACTUAL_POOL_PRICE(t) − price_lag_1`. Forecasting the delta rather than the level improves stationarity (mean ≈ 0, std ≈ 81.3 $/MWh) and prevents the model from learning a slowly drifting price baseline. |

---

### Cyclical Time Features

Extracted from `Date_Begin_Local` and encoded as sin/cos pairs to eliminate the discontinuity in raw integer encoding (e.g. hour 23 and hour 0 are 1 hour apart but 23 units apart as integers). Each pair encodes one full period on a unit circle.

| Columns | Period | Rationale |
|---|---|---|
| `hour_sin`, `hour_cos` | 24 hours | Intra-day price cycle (overnight low → morning ramp → evening peak). Most important time signal. |
| `dow_sin`, `dow_cos` | 7 days | Weekday vs. weekend industrial load difference. Complements `price_lag_168`. |
| `month_sin`, `month_cos` | 12 months | Residual seasonal variation: daylight hours, heating degree day accumulation, shoulder season dynamics. |
