# Alberta SMP Price Forecasting — Project TODO

## Goal
Forecast Alberta SMP pool price using two models: a baseline ARIMA and an LSTM. Compare performance to determine whether the added complexity of the LSTM is justified.

---

## Data

- [x] Fetch weather data (NASA POWER)
- [x] Join energy and weather datasets
- [x] Clean dataset (DST handling, dedup, NaN filling)
- [x] Drop plant-specific, export, and redundant columns
- [x] Engineer lag features
- [x] Engineer weather flags
- [x] Add cyclical time encodings
- [x] Define target variable (delta_price)

---

## Exploratory Data Analysis

- [ ] Distribution of SMP pool price and delta_price
- [ ] Seasonality analysis (hourly, daily, weekly, monthly patterns)
- [ ] Correlation analysis of features vs. target
- [ ] Visualise price spikes and cold snap / Chinook events
- [ ] Check stationarity of delta_price (ADF test)

---

## Baseline ARIMA Model

- [ ] Select and justify train / validation / test split
- [ ] Determine ARIMA order (p, d, q) using ACF / PACF plots
- [ ] Fit ARIMA on training set
- [ ] Evaluate on validation set (MAE, RMSE, MAPE)
- [ ] Tune order parameters
- [ ] Final evaluation on held-out test set
- [ ] Document results and limitations

---

## LSTM Model

- [ ] Select and justify train / validation / test split
- [ ] Normalise features (MinMax or Standard scaler — fit on train only)
- [ ] Define lookback window size
- [ ] Build LSTM architecture (layers, units, dropout)
- [ ] Train with early stopping
- [ ] Evaluate on validation set (MAE, RMSE, MAPE)
- [ ] Tune hyperparameters (learning rate, batch size, layers, units)
- [ ] Final evaluation on held-out test set
- [ ] Document results and limitations

---

## Model Comparison

- [ ] Compare ARIMA vs. LSTM on test set metrics
- [ ] Analyse where each model fails (price spikes, seasonal transitions)
- [ ] Determine whether LSTM complexity is justified over baseline
