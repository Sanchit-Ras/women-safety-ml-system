# Era A (2001-2014) honest baseline: 2026-09-13

Leak-free pipeline (`ml/features.py`, `ml/train.py`) on the 2001-2014 file only.
Units: 989. Windows: 4392. Features: 15 (lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, diff_1, diff_2, diff_3, diff_4, diff_5, lag_mean, lag_std, lag_last, lag_slope).
Target distribution: median 7.3%, p1 -66.7%, p99 174.7%, max 800%.

## Temporal split: train on label years < 2014, test on 2014 (568 windows)

| model | R2 | MAE | median AE | RMSE |
|---|---|---|---|---|
| xgboost | 0.061 | 23.0 | 16.6 | 37.4 |
| zero | -0.048 | 23.7 | 16.8 | 39.5 |
| persistence | -3.071 | 50.8 | 38.0 | 77.9 |
| window_mean | -0.224 | 26.9 | 20.3 | 42.7 |

## Grouped split: 80/20 by unit, seed 42 (900 test windows)

| model | R2 | MAE | median AE | RMSE |
|---|---|---|---|---|
| xgboost | 0.214 | 27.3 | 18.5 | 43.6 |
| zero | -0.084 | 29.1 | 18.2 | 51.2 |
| persistence | -1.516 | 46.7 | 28.5 | 78.0 |
| window_mean | -0.213 | 32.3 | 20.3 | 54.2 |

## XGBoost params

```json
{
  "n_estimators": 400,
  "max_depth": 4,
  "learning_rate": 0.03,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "min_child_weight": 5,
  "random_state": 42,
  "n_jobs": -1
}
```
