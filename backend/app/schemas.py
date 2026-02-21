from pydantic import BaseModel


class CrimeFeatures(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_5: float
    lag_6: float
    lag_mean: float
    lag_std: float
    lag_last: float
    lag_slope: float
