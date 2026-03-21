# LSTM Model — Design Decisions & Hyperparameter Log

Alberta SMP Pool Price Forecasting · PyTorch LSTM · `LSTM-Model.ipynb`

---

## 1. Problem Framing

**Target variable:** `delta_price` — the absolute hourly change in SMP pool price ($/MWh), defined as `ACTUAL_POOL_PRICE(t) − price_lag_1`.

Raw SMP is non-stationary: its mean and variance shift across years as generation mix, demand, and gas prices evolve. Forecasting the hourly change rather than the level keeps the target approximately stationary (mean ≈ 0, std ≈ 81.3 $/MWh), which is a prerequisite for stable LSTM training. The model predicts direction and magnitude of price movements, not the absolute price level.

**Evaluation metric:** RMSE and MAE against a naive baseline (predict zero change every hour). MAPE is excluded because near-zero values in `delta_price` produce arbitrarily large percentage errors that are not meaningful.

---

## 2. Data Splits

| Split | Rows | Period | Purpose |
|---|---|---|---|
| Train | ~31,000 | 2020-01-08 → 2024-01 | Model fitting |
| Validation | ~7,700 | 2024-01 → 2025-01 | Hyperparameter tuning, early stopping |
| Test | Separate file | 2025-01 → 2025-07 | Final unbiased evaluation (held out) |

Splits are **chronological** — no shuffling. Shuffling would create data leakage by allowing the model to see future states during training. The 80/20 train/validation split is applied to the `train.csv` file; the test set is a completely separate file that has never been used during model development.

> **Note on validation bias:** Hyperparameters were iteratively tuned using validation loss as feedback. This introduces mild model selection bias on the validation set — it is no longer a fully unbiased estimate of generalization. The held-out test set remains uncontaminated and is the authoritative metric for final model comparison against ARIMA.

---

## 3. Feature Set (25 input features)

### Continuous weather & demand
| Feature | Description |
|---|---|
| `T2M_AVG` | Province-average 2m air temperature |
| `WS50M_AVG` | Province-average 50m wind speed (hub height) |
| `RH2M_AVG` | Province-average relative humidity |
| `GHI_AVG` | Province-average solar irradiance |
| `T2M_CALGARY` | Calgary temperature (Chinook detection) |
| `WS50M_CALGARY` | Calgary wind speed (Chinook detection) |
| `T2M_LETHBRIDGE` | Lethbridge temperature (Chinook detection) |
| `WS50M_LETHBRIDGE` | Lethbridge wind speed (Chinook detection) |
| `ACTUAL_AIL` | Alberta Internal Load — total provincial demand |

### Lag features
| Feature | Lag | Rationale |
|---|---|---|
| `price_lag_1` | t−1h | Strongest autocorrelation; required to compute target |
| `price_lag_24` | t−24h | Same-hour-yesterday; intra-day seasonality |
| `price_lag_168` | t−168h | Same-hour-last-week; weekday vs. weekend regime |
| `AIL_lag_1` | t−1h | Demand shocks propagate within one settlement interval |
| `AIL_lag_24` | t−24h | Baseline demand comparison for anomaly detection |
| `T2M_AVG_lag_24` | t−24h | Temperature trend for `is_temp_dropping_fast` flag |
| `WS50M_AVG_lag_1` | t−1h | Wind ramp rate signal for generation forecasting |

### Binary weather flags (8 features)
| Flag | Condition | Samples | Rationale |
|---|---|---|---|
| `is_heating_season` | T2M_AVG < 10°C | 30,878 (63%) | Elevated heating demand regime |
| `is_cooling_season` | T2M_AVG > 20°C | 4,999 (10%) | A/C load in summer |
| `is_cold_snap` | T2M_AVG < −20°C | 1,257 (3%) | Demand surge + supply risk (gas freeze-offs) |
| `is_temp_dropping_fast` | ΔT24h < −8°C | 1,122 (2%) | Faster demand ramp than market anticipated |
| `is_low_wind` | WS50M_AVG < 4 m/s | 10,077 (21%) | Wind fleet near zero output, gas sets price |
| `is_high_wind` | WS50M_AVG > 12 m/s | 827 (2%) | Wind suppresses clearing price |
| `is_solar_generating` | GHI_AVG > 50 W/m² | 21,778 (45%) | Meaningful solar output online |
| `is_chinook` | ΔT_Calgary_24h > 10 + WS50M_Calgary > 8 + winter | 215 (0.4%) | Warm föhn winds; demand drop + wind surge |

### Cyclical time encodings (6 features)
`hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` — sin/cos pairs prevent the discontinuity problem of raw integer encoding (e.g., hour 23 and hour 0 are 1 hour apart, not 23). `Date_Begin_Local` is dropped before training.

---

## 4. Architecture Decisions

### Lookback window — 48 hours
The sliding window feeds the LSTM 48 consecutive hourly rows as one sequence. Each sequence is shaped `(48, 25)`.

- **Why 48 over 24:** A 24-hour window is the most common default, but Alberta's electricity market has a strong morning ramp / evening peak cycle — seeing two full days lets the model compare today's pattern against yesterday's without relying solely on the `price_lag_24` feature. The added context was found to improve validation loss.
- **At inference time:** Exactly 48 hours of recent observations are required to make one prediction. The window size does not change between training and deployment.

### Hidden size — 32
The LSTM's hidden state dimension. Reduced from an initial value of 64 after observing clear overfitting (training loss continued decreasing while validation loss diverged). A smaller hidden state has fewer parameters to overfit on a moderately sized dataset (~31k training sequences).

### Number of layers — 2
Two stacked LSTM layers allow the model to learn hierarchical temporal patterns: the first layer extracts short-range dependencies (hour-to-hour), the second layer can integrate these into longer-range patterns (daily cycles). Beyond 2 layers, gradient flow becomes harder to manage and the benefit is marginal for this dataset size.

### Dropout — 0.3
Applied both between LSTM layers (via `nn.LSTM(dropout=0.3)`) and before the final fully connected layer (`nn.Dropout(0.3)`). Increased from 0.2 after observing overfitting. Dropout is disabled during validation/inference via `model.eval()`.

> **Note:** `nn.LSTM(dropout=...)` only applies dropout between layers when `num_layers > 1`. For a single-layer LSTM it has no effect — the manual `nn.Dropout` after the last hidden state is always active regardless of layer count.

### Fully connected output layer
`nn.Linear(hidden_size=32, 1)` maps the final hidden state of the last LSTM layer to a single scalar (the predicted `delta_price`). `.squeeze(-1)` removes the trailing dimension to match the shape of `y_batch`.

**Total trainable parameters: 16,033** (down from 56,641 with hidden_size=64).

---

## 5. Scaling

`StandardScaler` is fit **only on the training set** and then applied to both validation and test sets. Fitting on the full dataset before splitting would constitute data leakage — the scaler would encode distributional information from future time periods into the training process.

`y` (delta_price) is **not scaled**. The loss and metrics remain in interpretable $/MWh units. Scaling the target is common but adds a de-scaling step at inference and makes the loss value harder to interpret against the naive baseline.

---

## 6. Loss Function — HuberLoss (δ = 10)

SMP `delta_price` has a heavy-tailed distribution — most hours are quiet (|Δprice| < 20 $/MWh), but price spike hours can exceed 400 $/MWh. MSELoss squares these large errors, causing the gradient to be dominated by rare spike events and making the model overfit to outliers at the expense of typical-hour accuracy.

**HuberLoss** behaves like MSE for errors below δ (quadratic, smooth gradients) and like MAE for errors above δ (linear, bounded gradient). With δ = 10:
- Errors under 10 $/MWh → quadratic penalty (fine-grained learning)
- Errors over 10 $/MWh → linear penalty (spike resistance)

This was changed from MSELoss after the initial training run showed the model being destabilized by spikes. The absolute loss values are lower with Huber than with MSE, so loss curves from different loss functions are not directly comparable.

---

## 7. Optimizer — Adam

Adam (Adaptive Moment Estimation) maintains per-parameter learning rates that adapt based on the first and second moments of the gradient history. It is the standard choice for LSTM training because:

- Handles sparse gradients well (some features activate rarely)
- Robust to noisy gradients from mini-batch training
- Converges faster than SGD with momentum on recurrent networks

**Learning rate: 0.001** — the Adam default. A lower rate (0.0005) can reduce oscillation during late training but slows convergence. A future experiment can test this if the current run shows val loss instability.

---

## 8. Scheduler — ReduceLROnPlateau

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

If validation loss does not improve for 10 consecutive epochs, the learning rate is halved. This prevents the optimizer from overshooting a flat region near the minimum and allows finer convergence without manually tuning a learning rate decay schedule.

- `factor=0.5` — halves the LR on each trigger
- `patience=10` — allows the model 10 epochs to escape a plateau before reducing

The scheduler monitors `epoch_val_loss`, the same signal as early stopping.

---

## 9. Early Stopping

```python
EarlyStopping(patience=15, min_delta=0)
```

Training halts if validation loss does not improve for 15 consecutive epochs. At the stopping point, `best_state` (a deep copy of model weights saved at the epoch of minimum validation loss) is restored via `model.load_state_dict(early_stopping.best_state)`.

**Why restore best_state and not final weights:** The final weights correspond to the most overfit state. The best_state corresponds to the epoch of optimal generalization — this is the stopping criterion, not the intersection of train and val loss curves.

**Patience selection:** patience=15 gives the scheduler (patience=10) at least one chance to reduce the LR and allow the model to escape a plateau before early stopping fires. Setting patience < scheduler patience would cause early stopping to trigger before the LR reduction has any effect.

---

## 10. Gradient Clipping

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Applied after `loss.backward()` and before `optimizer.step()`. Clips the global gradient norm to 1.0 if it exceeds that value. This guards against exploding gradients during backpropagation through time (BPTT), which can occur when the LSTM processes long sequences or encounters abrupt price spikes that produce very large loss values.

LSTMs mitigate the *vanishing* gradient problem internally via their gating mechanism (forget, input, output gates), but are not immune to *exploding* gradients — clipping remains a best practice.

---

## 11. Training Configuration Summary

| Hyperparameter | Value | Decision Basis |
|---|---|---|
| Lookback window | 48 hours | Two full days improves over 24h window |
| Batch size | 32 | Standard; balances gradient noise and memory |
| Hidden size | 32 | Reduced from 64 to reduce overfitting |
| LSTM layers | 2 | Hierarchical temporal feature extraction |
| Dropout | 0.3 | Increased from 0.2 after observing overfitting |
| Loss function | HuberLoss (δ=10) | Spike resistance vs. MSE |
| Optimizer | Adam | Standard for recurrent networks |
| Learning rate | 0.001 | Adam default |
| LR scheduler | ReduceLROnPlateau | factor=0.5, patience=10 |
| Early stopping | patience=15 | Fires after scheduler has had time to act |
| Gradient clipping | max_norm=1.0 | Exploding gradient guard |
| Max epochs | 150 | Upper bound; early stopping fires first |

---

## 12. Training Results (Current Run)

Early stopping triggered at **epoch 44**. Best validation loss recorded at **epoch 29** (val loss: 261.28 Huber units).

| Metric | LSTM | Naive Baseline (Δ=0) |
|---|---|---|
| Val MAE | ~30 $/MWh | ~31 $/MWh |
| Val RMSE | ~67 $/MWh | ~73 $/MWh |

The model modestly outperforms the naive baseline. Skill scores:
- **MAE skill: ~3%** above naive
- **RMSE skill: ~8%** above naive

This is expected for a first-pass LSTM on a dataset with high inherent noise. Alberta SMP is one of the most volatile electricity markets in North America due to its thermal-dominant generation mix, frequent price spikes, and relatively thin interconnections.

---

## 13. Known Limitations & Next Steps

- **Validation set is partially contaminated** by iterative hyperparameter tuning. Report final performance on the held-out test set only.
- **No exogenous supply data** (generation capacity outages, gas prices, merit order) — these are the primary drivers of price spikes and are not captured by weather or load alone.
- **Chinook flag is sparse** (215 samples, 0.4%). If ablation testing shows it adds noise rather than signal, it should be dropped.
- **Test set evaluation pending** — final RMSE/MAE on test set to be compared against ARIMA baseline.
- **Potential improvement areas:** attention mechanism over hidden states, additional price spike features (rolling volatility, spike history), or an ensemble of LSTM + ARIMA residual correction.
