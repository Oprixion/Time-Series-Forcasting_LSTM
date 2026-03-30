Accurate forecasting of Alberta's electricity spot market price (SMP) is critical for generators, retailers, and industrial consumers managing exposure to extreme price volatility. Existing univariate models struggle to anticipate sudden price spikes driven by compound events — cold snaps, wind droughts, and demand surges — that are characteristic of Alberta's deregulated market.

This study uses 5.5 years of hourly data (2020–2025) combining AESO pool price and load records with NASA POWER weather observations across three Alberta regions. After joining and cleaning the datasets, 17 features are engineered including price lags, a physics-grounded wind power proxy (WS50M³), a 24-hour rolling price volatility measure, temperature–demand interaction terms, and cyclical time encodings.

Three models are trained on a chronological 80/20 train/test split to forecast the hourly price change (delta_price): a univariate ARIMA(1,0,1) baseline, a LASSO regression, and a two-layer LSTM tuned via Bayesian optimisation (Optuna). Models are evaluated on overall RMSE and on conditional peak RMSE computed over the 99th percentile spike hours — the events of greatest practical interest.

EDA and feature analysis were contributed by Mena; ARIMA modelling by Mena and Tanvir; LASSO regression by Cynthia; LSTM architecture, tuning, and model comparison by Thaison.

On the held-out test set, LSTM achieves the lowest overall RMSE ([X] $/MWh) and the strongest spike capture performance (Peak RMSE: [X] $/MWh), outperforming both ARIMA ([X] $/MWh) and LASSO ([X] $/MWh) on extreme price events — demonstrating the value of sequential modelling for tail-risk forecasting in electricity markets.
