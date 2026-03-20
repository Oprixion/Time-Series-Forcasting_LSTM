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
**Shape:** 48,762 rows × 33 columns

---

### Base Features

| Column | Type | Description |
|---|---|---|
| `Date_Begin_Local` | datetime | Hour-beginning timestamp in local Alberta clock time. Reference index only — exclude from LSTM inputs. |
| `ACTUAL_POOL_PRICE` | float | System Marginal Price (SMP) in $/MWh. The raw target variable. |
| `ACTUAL_AIL` | float | Alberta Internal Load in MW. Total provincial electricity demand. |

---

### Weather Features

Per-city columns (Calgary, Lethbridge) are retained only for Chinook flag derivation. Province-wide averages carry the continuous weather signal for the model.

| Column | Type | Description |
|---|---|---|
| `T2M_CALGARY` | float | 2m air temperature in Calgary (°C). Used for Chinook detection. |
| `WS50M_CALGARY` | float | 50m wind speed in Calgary (m/s). Used for Chinook detection. |
| `T2M_LETHBRIDGE` | float | 2m air temperature in Lethbridge (°C). Used for Chinook detection. |
| `WS50M_LETHBRIDGE` | float | 50m wind speed in Lethbridge (m/s). Used for Chinook detection. |
| `T2M_AVG` | float | Province-average 2m air temperature (°C). |
| `WS50M_AVG` | float | Province-average 50m wind speed (m/s). Hub-height wind speed is the primary driver of wind generation output. |
| `RH2M_AVG` | float | Province-average relative humidity at 2m (%). |
| `GHI_AVG` | float | Province-average global horizontal irradiance (W/m²). Proxy for solar generation potential. |

---

### Lag Features

| Column | Type | Description |
|---|---|---|
| `price_lag_1` | float | SMP pool price 1 hour prior (t−1). Captures short-term autocorrelation. Also used to compute `delta_price`. |
| `price_lag_24` | float | SMP pool price 24 hours prior (t−24). Encodes same-hour-yesterday daily seasonality. |
| `price_lag_168` | float | SMP pool price 168 hours prior (t−168). Encodes same-hour-last-week weekly seasonality (weekday vs. weekend). |
| `AIL_lag_1` | float | Alberta Internal Load 1 hour prior (t−1). Captures immediate demand momentum. |
| `AIL_lag_24` | float | Alberta Internal Load 24 hours prior (t−24). Flags whether current demand is unusual relative to yesterday's norm. |
| `T2M_AVG_lag_24` | float | Province-average temperature 24 hours prior (t−24). Encodes the temperature trend direction for heating/cooling ramp detection. |
| `WS50M_AVG_lag_1` | float | Province-average 50m wind speed 1 hour prior (t−1). Captures wind generation ramp rate. |

---

### Weather Flags

Binary (0/1) features encoding discrete weather regimes with direct causal links to SMP behaviour.

| Column | True Samples | % | Condition | Rationale |
|---|---|---|---|---|
| `is_heating_season` | 30,878 | 63.3% | `T2M_AVG < 10°C` | Elevated heating demand drives baseline price upward for the majority of the Alberta year. |
| `is_cooling_season` | 4,999 | 10.3% | `T2M_AVG > 20°C` | AC load during Alberta's short summer creates a distinct warm-season price regime. |
| `is_cold_snap` | 1,257 | 2.6% | `T2M_AVG < -20°C` | Extreme cold simultaneously surges heating demand and risks gas freeze-offs, causing the sharpest price spikes in the dataset. |
| `is_temp_dropping_fast` | 1,122 | 2.3% | `T2M_AVG − T2M_AVG_lag_24 < −8°C` | Rapid cooling outpaces generator bid stacks, producing demand surprises that push prices up. |
| `is_low_wind` | 10,077 | 20.7% | `WS50M_AVG < 4 m/s` | Below turbine cut-in speed; Alberta's wind fleet drops to near-zero output, removing low-cost supply. |
| `is_high_wind` | 827 | 1.7% | `WS50M_AVG > 12 m/s` | Heavy wind generation suppresses clearing price, sometimes pushing toward zero off-peak. |
| `is_solar_generating` | 21,778 | 44.7% | `GHI_AVG > 50 W/m²` | Daytime solar output window. Mid-day solar suppression increasingly visible in the price curve as Alberta solar capacity has grown since 2020. |
| `is_chinook` | 215 | 0.4% | `(T2M_CALGARY − T2M_CALGARY_lag_24 > 10°C) & (WS50M_CALGARY > 8 m/s) & (month ∈ Oct–Mar)` | Warm föhn winds from the Rockies cause sudden winter demand collapse and wind surge simultaneously. Geographically concentrated in southern Alberta — uses Calgary per-city columns. Humidity condition dropped due to NASA POWER resolution limitations. |

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
