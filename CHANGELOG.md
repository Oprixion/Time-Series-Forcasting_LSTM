# Data Preparation Changelog

## V1 — Dataset Join & Cleaning Pipeline

The original workflow (`joinData.py` for weather fetching, `EDAJoin.ipynb` for joining) produced a 65 MB joined CSV with all 236 columns from the raw energy file. `prepare_dataset.py` replaces the join step with a single, reproducible script.

### Column Filtering

**Dropped ~225 plant-specific metered volume columns** (e.g. AFG1, AKE1, ALP1, BOW1, etc.). These are hourly generation volumes for individual power plants. They are not needed for a province-level SMP forecast — the pool price is already the market-clearing result of all generation bids, so feeding individual plant outputs back in would be redundant and noisy. If plant-level supply signals are needed later, they can be aggregated by fuel type as an engineered feature.

**Dropped 3 export columns** (EXPORT_BC, EXPORT_MT, EXPORT_SK). Exports represent power leaving Alberta and are less directly relevant to domestic price formation than imports, which add to Alberta's available supply and put downward pressure on the clearing price. Removing exports also reduces multicollinearity with the import columns.

**Kept 6 energy columns:**

- `ACTUAL_POOL_PRICE` — the SMP target variable
- `ACTUAL_AIL` — Alberta Internal Load, the province's total demand (correlation +0.22 with price)
- `HOUR_AHEAD_POOL_PRICE_FORECAST` — AESO's own forecast (correlation +0.91, strong baseline signal)
- `IMPORT_BC`, `IMPORT_MT`, `IMPORT_SK` — inter-provincial imports (correlations +0.07 to +0.20)

### DST Spring-Forward Fix

NASA POWER weather data uses true MST (no daylight saving), so it contains 02:00 rows on spring-forward Sundays. AESO energy data uses clock time, which skips from 01:00 to 03:00 on those days. The script remaps the 6 affected weather rows (one per year, 2020–2025) from 02:00 → 03:00 before joining, preventing a timestamp mismatch that would create NaN rows.

### DST Fall-Back Dedup

On fall-back Sundays in November, AESO records two separate rows for the same clock hour (two 01:00s and two 02:00s). This creates 10 duplicate timestamps across the dataset. Because an LSTM requires a strictly monotonic time index, the duplicate pairs are averaged into single rows. This preserves the true energy values (the average of the two settlement periods) while maintaining a clean sequential structure.

### Missing Weather Data Handling

The weather dataset starts at 2020-01-01 01:00 and ends at 2025-07-31 18:00, while the energy dataset covers 2020-01-01 00:00 to 2025-07-31 23:00. This leaves 6 edge-case rows (1 at the start, 5 at the end) with no weather match. These are filled using forward-fill then back-fill — appropriate here because weather conditions change slowly relative to hourly resolution, and the affected rows are at the dataset boundaries.

---

## V2 — Feature Engineering for LSTM Forecasting

Building on the clean joined dataset from V1, the pipeline now engineers lag features and weather flags, then defines the forecasting target (hourly absolute price change). The output grows from 27 to 31 columns while the row count drops slightly to 48,762 due to lag warm-up.

### Redundant Weather Column Removal

**Dropped all Edmonton per-city columns** (T2M_EDMONTON, WS10M_EDMONTON, WS50M_EDMONTON, RH2M_EDMONTON, ALLSKY_SFC_SW_DWN_EDMONTON). Edmonton's values are captured by the provincial averages (T2M_AVG, WS50M_AVG, etc.), and the SMP is a province-level clearing price — individual city weather only earns its place if it enables a geographically specific signal like Chinook detection.

**Dropped all WS10M columns** (WS10M_CALGARY, WS10M_LETHBRIDGE, WS10M_AVG). Surface-level 10m wind speed is redundant with WS50M (50m hub-height wind speed), which is the physically relevant measurement for wind turbine generation output. WS50M is what drives the supply-side effect on SMP.

**Dropped per-city solar irradiance** (ALLSKY_SFC_SW_DWN_CALGARY, ALLSKY_SFC_SW_DWN_LETHBRIDGE). Province-average GHI_AVG captures the solar generation signal. Alberta's solar farms are distributed, so a single average is sufficient for a province-level price model.

**Dropped per-city relative humidity** (RH2M_CALGARY, RH2M_LETHBRIDGE). Initially kept for Chinook detection, but NASA POWER's coarse spatial resolution (~0.5° grid) fails to capture the characteristic humidity drop during Chinook events — mean RH during identified Chinook conditions was still ~80%. RH2M_AVG is retained for general humidity signal.

**Kept Calgary and Lethbridge temperature and wind** (T2M_CALGARY, WS50M_CALGARY, T2M_LETHBRIDGE, WS50M_LETHBRIDGE). These are required for Chinook detection, which is geographically concentrated in southern Alberta. Edmonton does not experience Chinooks, so the provincial average would dilute the signal.

### Lag Features (7 features)

Lag features provide the LSTM with explicit lookback signals that may fall outside its input window. The LSTM's gating mechanism learns sequential dependencies within the lookback window, but lags at 24h and 168h capture daily and weekly seasonality that the window may not span.

**Price lags:**

- `price_lag_1` (t−1) — Previous hour's SMP. Strongest autocorrelation in the series. Also required to compute the delta_price target.
- `price_lag_24` (t−24) — Same hour yesterday. Electricity markets have strong intra-day seasonality (morning ramp, evening peak, overnight trough). This lets the model compare "is the current price high or low relative to this time yesterday?"
- `price_lag_168` (t−168) — Same hour last week. Weekday vs. weekend load patterns are very different in Alberta (industrial demand drops on weekends). This is the most commonly cited lag in electricity price forecasting literature.

**Demand lags:**

- `AIL_lag_1` (t−1) — Previous hour's Alberta Internal Load. Demand shocks propagate into price within one settlement interval. A sudden load increase means more expensive generators must be dispatched.
- `AIL_lag_24` (t−24) — Same hour yesterday's load. Helps the model distinguish whether current demand is unusual relative to the daily norm — an abnormally high load at 3 AM signals something different than a high load at 6 PM.

**Weather lags:**

- `T2M_AVG_lag_24` (t−24) — Temperature 24 hours ago. Temperature-driven demand (heating/cooling) has a delayed effect as buildings respond thermally. The difference between current and lagged temperature also encodes the warming/cooling trend, which is critical for the `is_temp_dropping_fast` flag.
- `WS50M_AVG_lag_1` (t−1) — Wind speed one hour ago. Wind generation responds almost instantly to wind speed changes, and generators incorporate recent wind ramp rates into their bidding strategies. Only lag-1 is needed since wind is not periodic like temperature.

Total rows lost to lag warm-up: 168 (the maximum lag). Trivial against 48,930 rows.

### Weather Flags (8 binary features)

Each flag encodes a specific physical regime with a direct causal link to SMP price behaviour. Binary encoding is used rather than continuous thresholds because the LSTM already has the continuous values — the flags mark discrete regime boundaries where price behaviour changes qualitatively.

**Temperature flags:**

- `is_heating_season` — `T2M_AVG < 10°C` — **30,878 samples (63.3%)**. Alberta's primary demand driver. When temperatures are below 10°C, residential and commercial heating load elevates baseline demand, which pushes the merit order into more expensive generation. Active for roughly half the year, reflecting Alberta's northern climate.

- `is_cooling_season` — `T2M_AVG > 20°C` — **4,999 samples (10.3%)**. Air conditioning load during Alberta's short summer. A smaller effect than heating (Alberta's cooling demand is modest compared to southern jurisdictions), but still produces a distinct price regime during July–August heat events.

- `is_cold_snap` — `T2M_AVG < -20°C` — **1,257 samples (2.6%)**. Extreme cold events where two price-driving forces compound: heating demand surges to annual peaks AND natural gas freeze-offs can knock generation offline, tightening supply. These events produce the most extreme price spikes in the dataset. Separated from `is_heating_season` because the price dynamics are qualitatively different (supply risk, not just demand elevation).

- `is_temp_dropping_fast` — `(T2M_AVG − T2M_AVG_lag_24) < −8°C` — **1,122 samples (2.3%)**. Captures rapid cooling events where demand ramps faster than the market anticipated when generators placed their bids. A temperature that is −15°C and falling is a different price environment than −15°C and stable, even though the current reading is identical.

**Wind flags:**

- `is_low_wind` — `WS50M_AVG < 4 m/s` — **10,077 samples (20.7%)**. Below typical turbine cut-in speed (~3.5–4.5 m/s), Alberta's wind fleet generates near-zero output. This removes a significant low-marginal-cost supply source from the merit order, allowing more expensive gas generators to set the price. Occurs roughly 1 in 5 hours.

- `is_high_wind` — `WS50M_AVG > 12 m/s` — **827 samples (1.7%)**. Heavy wind generation suppresses the clearing price by displacing expensive thermal generation. During off-peak hours combined with high wind, prices can approach zero or even go negative. The −0.28 correlation between WS50M_AVG and price is the strongest weather-price relationship in the dataset.

**Solar flag:**

- `is_solar_generating` — `GHI_AVG > 50 W/m²` — **21,778 samples (44.7%)**. Marks hours when solar irradiance is sufficient for meaningful photovoltaic output. Alberta's solar capacity has grown significantly since 2020, and mid-day solar suppression is increasingly visible in the price curve. The 50 W/m² threshold filters out dawn/dusk hours where irradiance exists but generation is negligible.

**Chinook flag:**

- `is_chinook` — `(T2M_CALGARY − T2M_CALGARY_lag_24 > 10) & (WS50M_CALGARY > 8) & (winter months Oct–Mar)` — **215 samples (0.4%)**. Chinook winds are a uniquely Alberta phenomenon: warm, dry föhn winds from the Rockies that can raise Calgary/Lethbridge temperatures by 15–25°C in hours during winter. The price impact is a sudden demand drop (heating load collapses) combined with increased wind generation — both pushing prices down sharply during what the model otherwise expects to be a high-demand winter period.

  Detection uses Calgary-specific columns because Chinooks are geographically concentrated in southern Alberta. The original detection criteria included `RH2M_CALGARY < 40%`, but NASA POWER's ~0.5° grid resolution does not capture the characteristic humidity drop — mean RH during identified events was 80%, not the expected <40%. The flag was recalibrated to use temperature rise + wind speed + winter seasonality, yielding 215 samples representing approximately 20–30 distinct Chinook events over 5.5 years, consistent with observed Chinook frequency in the region.

  At 0.4% of the dataset, this is the thinnest flag. If it introduces noise rather than signal during LSTM training, it can be dropped without affecting the other features.

### Target Variable

- `delta_price` — `ACTUAL_POOL_PRICE(t) − price_lag_1` — The absolute hourly change in SMP pool price. Forecasting the change rather than the level is preferred for LSTM because raw SMP is highly non-stationary (mean and variance shift across years), while the hourly delta is approximately stationary (mean ≈ 0, std ≈ 81.3). This avoids the model learning a slowly drifting baseline and focuses it on predicting the direction and magnitude of price movements.

### Cyclical Time Features (6 features)

`Date_Begin_Local` is decomposed into three periodic components, each encoded as a sin/cos pair rather than a raw integer.

Raw integer encoding creates a discontinuity problem — hour 23 and hour 0 are numerically 23 units apart but only 1 hour apart in real time. The LSTM would have to learn this wrap-around relationship from scratch. Sin/cos encoding maps each periodic value onto a unit circle, so the Euclidean distance between any two encoded values correctly reflects their temporal proximity. For example, 11 PM and midnight are placed adjacent in feature space, which is where they belong.

- `hour_sin` / `hour_cos` — hour of day, period 24. The most important time encoding. Electricity prices have a strong and consistent intra-day cycle: low overnight, morning ramp as industrial load comes online, mid-day plateau, evening peak as residential demand adds to commercial load, then overnight trough again. The LSTM benefits from knowing explicitly where in the day each row sits rather than inferring it from the sequence alone.

- `dow_sin` / `dow_cos` — day of week, period 7 (Monday = 0). Alberta's industrial load drops significantly on weekends, which flattens the daily demand curve and lowers average prices. A Friday evening and a Saturday evening can have identical temperature and wind readings but very different price behaviour. The day-of-week encoding lets the model distinguish these without relying entirely on price_lag_168 to carry the weekly pattern.

- `month_sin` / `month_cos` — month of year, period 12. Captures residual seasonal variation not already explained by the temperature and weather flags — for example, daylight hours (which affect when solar generates and when lighting load peaks), heating degree day accumulation across a winter, and spring/fall shoulder season price dynamics.

`Date_Begin_Local` is retained in the dataframe as a reference index for tracing predictions back to specific timestamps, but must be excluded from the LSTM input feature array during training.

### Final Output

`Data/energy_weather_featured.csv` — 48,762 rows × 37 columns (15 continuous features, 7 lag features, 8 binary flags, 6 cyclical time encodings, 1 target), zero NaN values, monotonically increasing timestamps from 2020-01-08 to 2025-07-31. Ready for train/validation/test splitting and LSTM training.
