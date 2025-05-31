import numpy as np
import pandas as pd
import cvxpy as cp

# --- Configuration ---
N_ASSETS = 10
N_USERS = 120
N_DAYS = 10
PRICE_START = 100.0
SEED = 2024
DAILY_VOL_RANGE = (0.006, 0.02)
DAILY_MU_MEAN = 0.05/252
DAILY_MU_STD = 0.015
USR_SIG_RANGE = (0.07, 0.25)
CAP_RANGE = (10_000, 150_000)
np.random.seed(SEED)

# --- Market Data Simulation ---
asset_names = [f"A{i+1}" for i in range(N_ASSETS)]
true_sigma = np.random.uniform(*DAILY_VOL_RANGE, N_ASSETS)
true_mu = np.random.normal(DAILY_MU_MEAN, DAILY_MU_STD, N_ASSETS)
A = np.random.randn(N_ASSETS, N_ASSETS)
Σ = A @ A.T
D = np.diag(true_sigma / np.sqrt(np.diag(Σ)))
Σ = D @ Σ @ D
daily_rets = np.random.multivariate_normal(true_mu, Σ, size=N_DAYS)
price_series = PRICE_START * np.exp(np.cumsum(daily_rets, axis=0))

# --- User Data ---
σ_budget = np.random.uniform(*USR_SIG_RANGE, N_USERS)
capacity = np.random.uniform(*CAP_RANGE, N_USERS)
weights_raw = np.random.dirichlet(np.ones(N_ASSETS), size=N_USERS)
mask_zero = np.random.rand(N_USERS, N_ASSETS) < 0.60
weights_raw[mask_zero] = 0
weights_norm = weights_raw / weights_raw.sum(axis=1, keepdims=True)
weights_norm = np.nan_to_num(weights_norm)
users_df = pd.DataFrame({"User": [f"U{i+1}" for i in range(N_USERS)],
                         "σ_budget": σ_budget,
                         "capacity": capacity})

# --- Portfolio Builders ---
def normalise(w):
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1/len(w))

def mvo_weights(mu, Σ, σ_cap):
    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(mu @ w),
                      [cp.sum(w)==1, w>=0, cp.quad_form(w, Σ)<=σ_cap**2])
    prob.solve(solver=cp.SCS, verbose=False)
    return normalise(w.value) if w.value is not None else np.ones(n)/n

inv_vol = 1/np.sqrt(np.diag(Σ))
rp_base = inv_vol / inv_vol.sum()
def rp_weights(Σ, σ_cap):
    base_vol = np.sqrt(rp_base.T @ Σ @ rp_base)
    scale = min(σ_cap / base_vol, 1)
    return normalise(rp_base * scale)

def ew_weights(n): return np.ones(n)/n

# --- Simulation Loop ---
records = []
metrics = []

for day in range(N_DAYS):
    daily_mu = daily_rets[day]
    future_returns = daily_mu  # Simplified proxy for next-day return
    
    for idx, u in users_df.iterrows():
        σ_cap = u["σ_budget"]
        cap = u["capacity"]
        w_mvo = mvo_weights(daily_mu, Σ, σ_cap)
        w_rp = rp_weights(Σ, σ_cap)
        w_ew = ew_weights(N_ASSETS)
        original_portfolio = weights_norm[idx]
        original_return = np.dot(original_portfolio, future_returns)
        user_assets = set([a for a, w in zip(asset_names, original_portfolio) if w > 0])

        for name, w in [("MVO", w_mvo), ("Risk Parity", w_rp), ("Equal Weight", w_ew)]:
            exp_ret = np.dot(w, daily_mu)
            delta_r = np.dot(w, future_returns) - original_return
            improved_r = int(delta_r > 0)
            ranked_assets = [a for _, a in sorted(zip(w, asset_names), reverse=True)]

            for k in [3,5]:
                top_k = set(ranked_assets[:k])
                hit = int(len(top_k & user_assets) > 0)
                metrics.append({
                    "Day": day+1, "User": u["User"], "Strategy": name, "k": k,
                    "HR": hit, "P(R)": improved_r, "ΔR": delta_r
                })
            records.append({
                "Day": day+1, "User": u["User"], "Strategy": name,
                "ROI_percent": round(exp_ret,6),
                "ROI_dollars": round(exp_ret*cap,2)
            })

# --- Results Aggregation ---
results_df = pd.DataFrame(records)
metrics_df = pd.DataFrame(metrics)

agg_metrics_overall = metrics_df.groupby(["Strategy", "k"]).agg(
    HR=("HR","mean"), P_R=("P(R)","mean"), ΔR=("ΔR","mean")
).reset_index()

avg_results = results_df.groupby("Strategy").agg(
    Avg_ROI_percent=("ROI_percent", "mean"),
    Avg_ROI_dollars=("ROI_dollars", "mean")
).reset_index()

# --- Print Results ---
print("\n--- Average Portfolio Results Across All Users ---\n", avg_results)
print("\n--- Aggregated Metrics (HR@k, P(R)@k, ΔR@k) ---\n", agg_metrics_overall)
