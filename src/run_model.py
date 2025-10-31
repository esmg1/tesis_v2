import argparse
import os
from pathlib import Path

import pandas as pd

from data_utils import ensure_curated_events

def load_curated():
    p = ensure_curated_events()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df

def run_dstm(df):
    import numpy as np
    import arviz as az
    import pymc as pm

    Path("artifacts").mkdir(exist_ok=True)
    # Aggregate by (region, month) as a simple starter; you can swap to daily.
    g = df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp()) \
          .groupby(["region","month"], as_index=False)["y"].sum()
    # Encode regions
    regions = g["region"].unique()
    r_index = g["region"].astype("category").cat.codes.values
    t, t_index = np.unique(g["month"].values, return_inverse=True)
    R, T = len(regions), len(t)
    y = g["y"].astype("int64").values

    with pm.Model() as m:
        # Region intercepts (weak spatial prior starter — independent; replace with CAR later)
        alpha = pm.Normal("alpha", 0.0, 1.0, shape=R)

        # Temporal component: AR(1) on latent log-intensity shared across regions
        rho = pm.Uniform("rho", -0.95, 0.95)
        sigma_t = pm.Exponential("sigma_t", 1.0)
        z = pm.AR("z", rho=[rho], sigma=sigma_t, shape=T)

        # Overdispersion
        phi = pm.Exponential("phi", 1.0)  # negbin concentration

        eta = alpha[r_index] + z[t_index]  # log mean
        mu = pm.math.exp(eta)
        pm.NegativeBinomial("y", mu=mu, alpha=phi, observed=y)

        idata = pm.sample(
            200,
            tune=200,
            target_accept=0.99,
            chains=2,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )
        ppc = pm.sample_posterior_predictive(idata, var_names=["y"], random_seed=43)
        idata.extend(ppc)

        az.to_netcdf(idata, "artifacts/dstm_idata.nc")

        # Posterior predictive mean for observed period
        y_ppc = ppc.posterior_predictive["y"].mean(("chain", "draw")).values
        g["y_hat"] = y_ppc
        g.to_parquet("artifacts/dstm_forecast.parquet")
    return "artifacts/dstm_forecast.parquet"

def run_esn(df):
    import numpy as np, json
    # Simple univariate ESN on national aggregate as a quick deep baseline
    s = df.sort_values("date").groupby("date", as_index=False)["y"].sum()
    y = s["y"].values.astype(float)
    # Train/test split (last 10% test)
    n = len(y); split = int(n*0.9)
    y_tr, y_te = y[:split], y[split:]

    # Minimal ESN
    rng = np.random.default_rng(1)
    N = 400; sr = 0.9; leak=0.3; ridge=1e-2
    W = rng.standard_normal((N,N)); mask = rng.random((N,N)) < 0.05; W *= mask
    # Spectral radius normalization
    eig = max(abs(np.linalg.eigvals(W))); W *= (sr / eig)
    Win = rng.standard_normal((N,1))*0.1
    state = np.zeros((N,1))
    X=[]; Y=[]
    for v in y_tr:
        state = (1-leak)*state + leak*np.tanh(W@state + Win*v)
        X.append(state.ravel()); Y.append([v])
    X = np.vstack(X); Y = np.vstack(Y)
    # ridge regression
    Wout = np.linalg.solve(X.T@X + ridge*np.eye(N), X.T@Y)

    # Predict
    preds=[]
    state = np.zeros((N,1))
    last = y_tr[-1]
    for _ in range(len(y_te)):
        state = (1-leak)*state + leak*np.tanh(W@state + Win*last)
        yhat = float(state.ravel()@Wout)
        preds.append(max(0.0, yhat))
        last = yhat
    os.makedirs("artifacts", exist_ok=True)
    out = s.iloc[split:].copy()
    out["y_hat"] = preds
    out.to_parquet("artifacts/esn_forecast.parquet")
    return "artifacts/esn_forecast.parquet"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dstm","esn"], required=True)
    args = ap.parse_args()
    df = load_curated()
    path = run_dstm(df) if args.model=="dstm" else run_esn(df)
    print(path)
