import json, pandas as pd, numpy as np, arviz as az, os
from pathlib import Path

def rmse_mae(df, y="y", yhat="y_hat"):
    e = df[yhat] - df[y]
    return {"rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e)))}

def eval_dstm():
    import xarray as xr
    metrics={}
    if Path("artifacts/dstm_idata.nc").exists():
        idata = az.from_netcdf("artifacts/dstm_idata.nc")
        waic = az.waic(idata).elpd_waic
        metrics["waic"] = float(waic)
    if Path("artifacts/dstm_forecast.parquet").exists():
        d = pd.read_parquet("artifacts/dstm_forecast.parquet")
        metrics |= rmse_mae(d, y="y", yhat="y_hat")
    return metrics

def eval_esn():
    if Path("artifacts/esn_forecast.parquet").exists():
        d = pd.read_parquet("artifacts/esn_forecast.parquet")
        return rmse_mae(d, y="y", yhat="y_hat")
    return {}

if __name__ == "__main__":
    Path("artifacts").mkdir(exist_ok=True)
    metrics = {"dstm": eval_dstm(), "esn": eval_esn()}
    with open("artifacts/metrics.json","w") as f: json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
