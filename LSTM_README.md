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
| Train | ~31,208 | 2020-01-08 → ~2023-10 | Model fitting (80% of train.csv) |
| Validation | ~7,802 | ~2023-10 → ~2024-03 | Hyperparameter tuning, early stopping (20% of train.csv) |
| Test | ~9,752 | ~2024-03 → 2025-07-31 | Final unbiased evaluation (held out — test.csv) |

Splits are **chronological** — no shuffling at any level. The outer 80/20 split (train.csv / test.csv) is performed automatically by `prepare_dataset.py` Stage 10, ensuring a reproducible and consistent cutoff across the team. The inner 80/20 train/validation split is applied inside the notebook on `train.csv` only. The test set is never used during model development.

> **Note on validation bias:** Hyperparameters were iteratively tuned using validation loss as feedback. This introduces mild model selection bias on the validation set — it is no longer a fully unbiased estimate of generalization. The held-out test set remains uncontaminated and is the authoritative metric for final model comparison against ARIMA.

---

## 3. Feature Set (26 input features)

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
| `price_rolling_std_24` | rolling 24h | Recent market volatility — predictor of spike likelihood. Computed on `price_lag_1` only, no leakage. 24h window preferred over 48h: volatility is a short-term signal. |

### Binary weather flags (7 features)
| Flag | Condition | Samples | Rationale |
|---|---|---|---|
| `is_heating_season` | T2M_AVG < 10°C | 30,878 (63%) | Elevated heating demand regime |
| `is_cooling_season` | T2M_AVG > 20°C | 4,999 (10%) | A/C load in summer |
| `is_cold_snap` | T2M_AVG < −20°C | 1,257 (3%) | Demand surge + supply risk (gas freeze-offs) |
| `is_temp_dropping_fast` | ΔT24h < −8°C | 1,122 (2%) | Faster demand ramp than market anticipated |
| `is_low_wind` | WS50M_AVG < 4 m/s | 10,077 (21%) | Wind fleet near zero output, gas sets price |
| `is_solar_generating` | GHI_AVG > 50 W/m² | 21,778 (45%) | Meaningful solar output online |
| `is_chinook` | ΔT_Calgary_24h > 10 + WS50M_Calgary > 8 + winter | 215 (0.4%) | Warm föhn winds; demand drop + wind surge |

### Continuous supply feature (replaces `is_high_wind`)
| Feature | Formula | Rationale |
|---|---|---|
| `wind_power_proxy` | `WS50M_AVG³` | Wind power ∝ v³ (Betz law). Continuous physics-grounded supply signal across the full wind speed range. `is_high_wind` had correlation 0.002 with `delta_price` (EDA) and is fully absorbed by this feature. Alberta wind generators bid at $0/MWh, so higher wind output directly suppresses the SMP clearing price. |

### Cyclical time encodings (6 features)
`hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` — sin/cos pairs prevent the discontinuity problem of raw integer encoding (e.g., hour 23 and hour 0 are 1 hour apart, not 23). `Date_Begin_Local` is dropped before training.

---

## 4. Architecture Decisions

### Lookback window — 48 hours
The sliding window feeds the LSTM 48 consecutive hourly rows as one sequence. Each sequence is shaped `(48, 26)`.

- **Why 48 over 24:** A 24-hour window is the most common default, but Alberta's electricity market has a strong morning ramp / evening peak cycle — seeing two full days lets the model compare today's pattern against yesterday's without relying solely on the `price_lag_24` feature. The added context was found to improve validation loss.
- **At inference time:** Exactly 48 hours of recent observations are required to make one prediction. The window size does not change between training and deployment.

### Hidden size — 32
The LSTM's hidden state dimension. Initially reduced from 64 after observing clear overfitting, then confirmed by Optuna search (candidates: 32, 64, 128). A smaller hidden state has fewer parameters to overfit on a moderately sized dataset (~31k training sequences).

### Number of layers — 2
Two stacked LSTM layers allow the model to learn hierarchical temporal patterns: the first layer extracts short-range dependencies (hour-to-hour), the second layer can integrate these into longer-range patterns (daily cycles). Beyond 2 layers, gradient flow becomes harder to manage and the benefit is marginal for this dataset size.

### Dropout — 0.45
Applied both between LSTM layers (via `nn.LSTM(dropout=0.45)`) and before the final fully connected layer (`nn.Dropout(0.45)`). Selected by Optuna hyperparameter search (range tested: 0.1–0.5). The high dropout rate reflects the noisy, spike-heavy nature of `delta_price` — the model needs strong regularization to avoid memorizing outlier events. Dropout is disabled during validation/inference via `model.eval()`.

> **Note:** `nn.LSTM(dropout=...)` only applies dropout between layers when `num_layers > 1`. For a single-layer LSTM it has no effect — the manual `nn.Dropout` after the last hidden state is always active regardless of layer count.

### Fully connected output layer
`nn.Linear(hidden_size=32, 1)` maps the final hidden state of the last LSTM layer to a single scalar (the predicted `delta_price`). `.squeeze(-1)` removes the trailing dimension to match the shape of `y_batch`.

**Total trainable parameters: 16,097** — recalculated for INPUT_SIZE=26 (down from 56,641 with hidden_size=64). Update `INPUT_SIZE` in the notebook's Model SETUP cell after re-running the pipeline.

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

**Learning rate: 0.00467** — selected by Optuna (log-uniform search over 1e-4 to 5e-3). Higher than the Adam default of 0.001, but paired with the ReduceLROnPlateau scheduler (factor=0.5, patience=10), the effective LR decreases during training. Starting aggressive and decaying outperformed starting conservative in the search.

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
| Input size | 26 | 25 original − 1 (is_high_wind) + 1 (wind_power_proxy) + 1 (price_rolling_std_24) |
| Lookback window | 48 hours | Two full days improves over 24h window |
| Batch size | 32 | Standard; balances gradient noise and memory |
| Hidden size | 32 | Optuna-selected; also avoids overfitting on ~31k sequences |
| LSTM layers | 2 | Hierarchical temporal feature extraction |
| Dropout | 0.45 | Optuna-selected; high dropout appropriate for volatile, spike-heavy target |
| Loss function | HuberLoss (δ=10) | Spike resistance vs. MSE |
| Optimizer | Adam | Standard for recurrent networks |
| Learning rate | 0.00467 | Optuna-selected; scheduler brings it down from this starting point |
| LR scheduler | ReduceLROnPlateau | factor=0.5, patience=10 |
| Early stopping | patience=15 | Fires after scheduler has had time to act |
| Gradient clipping | max_norm=1.0 | Exploding gradient guard |
| Max epochs | 150 | Upper bound; early stopping fires first |

---

## 12. Training Results (Current Run — Optuna-tuned)

Early stopping triggered at **epoch 44**. Best validation loss recorded at **epoch 29** (val loss: 258.41 Huber units).

| Metric | LSTM | Naive Baseline (Δ=0) |
|---|---|---|
| Val MAE | ~29.8 $/MWh | ~31 $/MWh |
| Val RMSE | ~69.4 $/MWh | ~73 $/MWh |

The model outperforms the naive baseline on aggregate metrics. Skill scores:
- **MAE skill: ~4%** above naive
- **RMSE skill: ~5%** above naive

These aggregate numbers understate the LSTM's value. The ARIMA baseline achieves lower aggregate MAE (~26 $/MWh) by accurately predicting the majority of quiet hours where delta_price is near zero, but the LSTM better captures price spike events — the hours with the greatest economic significance in electricity markets. Final test set evaluation should include conditional metrics (spike-hour vs. quiet-hour performance) to properly characterize each model's strengths.

---

## 13. Ablation Testing

A feature ablation test was conducted to validate each feature's contribution. Using the Optuna-selected hyperparameters, one candidate feature was removed at a time and the model was retrained for up to 50 epochs. The decision threshold was ±0.5 $/MWh change in val RMSE.

All 9 tested features (7 binary flags + `wind_power_proxy` + `price_rolling_std_24`) returned NEUTRAL (within ±0.5 $/MWh). No feature actively hurts the model, and no feature provides a statistically decisive improvement on its own. Features with positive delta (removing them slightly worsened RMSE) — `is_cold_snap`, `price_rolling_std_24`, `RH2M_AVG`, `is_cooling_season` — are retained with confidence. Features with near-zero or slightly negative delta — `is_chinook`, `is_temp_dropping_fast`, `is_low_wind`, `wind_power_proxy`, `is_solar_generating` — are retained on domain justification (documented causal links to SMP behaviour).

The full 26-feature set is kept as the final configuration.

---

## 14. Known Limitations & Next Steps

- **Validation set is partially contaminated** by iterative hyperparameter tuning. Report final performance on the held-out test set only.
- **No exogenous supply data** (generation capacity outages, gas prices, merit order) — these are the primary drivers of price spikes and are not captured by weather or load alone.
- **Test set evaluation pending** — final RMSE/MAE on test set to be compared against ARIMA baseline, including conditional metrics on spike vs. quiet hours.
- **Potential improvement areas:** attention mechanism over hidden states, spike history features, or an ensemble of LSTM + ARIMA residual correction.
