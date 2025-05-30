# src/heuristics/utils.py
import numpy as np
import pandas as pd
import logging
from typing import Tuple, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from environment.financial_env import FinancialRecEnv  # Assuming flat src structure

log = logging.getLogger(__name__)


def normalise_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalizes a weight vector to sum to 1, ensuring non-negativity.
    If all weights are zero or negative, returns an equal weight distribution.
    """
    # Ensure non-negative weights
    processed_weights = np.maximum(weights, 0)
    sum_weights = np.sum(processed_weights)

    if sum_weights > 1e-9:  # Use a small epsilon for numerical stability
        return processed_weights / sum_weights
    else:
        # Fallback to equal weights if sum is zero (e.g., all inputs were <= 0)
        log.debug(
            "Sum of weights is near zero in normalise_weights; falling back to equal weights."
        )
        num_assets = len(weights)
        return np.full(num_assets, 1.0 / num_assets) if num_assets > 0 else np.array([])


def calculate_mu_sigma_hat(
    prices_df_wide: pd.DataFrame,  # Expects DatetimeIndex, ISINs as columns
    current_simulation_date: pd.Timestamp,
    lookback_days: int,
    num_total_assets_in_env: int,  # For fallback if prices_df_wide is empty
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates mu_hat (expected daily log returns) and Sigma_hat (covariance of daily log returns)
    from historical daily prices up to the current_simulation_date.

    Args:
        prices_df_wide: DataFrame with DatetimeIndex and asset ISINs as columns, values are close prices.
        current_simulation_date: The current date in the simulation (end of the lookback period).
        lookback_days: Number of trading days of price history to use.
        num_total_assets_in_env: Total number of assets in the environment, used for default return shapes.

    Returns:
        Tuple (mu_hat, sigma_hat). mu_hat is 1D array, sigma_hat is 2D array.
    """
    if (
        prices_df_wide.empty or prices_df_wide.shape[1] == 0
    ):  # No assets in prices_df_wide
        log.warning(
            f"calculate_mu_sigma_hat: prices_df_wide is empty or has no asset columns. "
            f"Returning default zero forecast for {num_total_assets_in_env} assets."
        )
        return np.zeros(num_total_assets_in_env), np.eye(num_total_assets_in_env) * 1e-6

    n_assets_in_prices = prices_df_wide.shape[1]

    # Determine the date range for historical prices
    # We need 'lookback_days' number of price points.
    end_date_dt = pd.to_datetime(current_simulation_date)
    # To get 'lookback_days' points, we need to go back 'lookback_days - 1' from the end_date.
    # However, market data might have gaps (weekends, holidays).
    # So, we take a slightly larger window and then .tail(lookback_days).
    # A common approach is to take a window like 1.5*lookback_days calendar days.
    approx_start_date = end_date_dt - pd.Timedelta(
        days=int(lookback_days * 1.5) + 5
    )  # Generous window

    # Filter prices for the lookback window. prices_df_wide should already be sorted by date.
    price_history_slice = prices_df_wide[
        (prices_df_wide.index >= approx_start_date)
        & (prices_df_wide.index <= end_date_dt)
    ].tail(lookback_days)  # Get the most recent 'lookback_days' available data points

    if len(price_history_slice) < lookback_days:
        log.warning(
            f"Not enough price history ({len(price_history_slice)} days) for lookback ({lookback_days}) "
            f"ending on {end_date_dt.strftime('%Y-%m-%d')}. Data available from {price_history_slice.index.min() if not price_history_slice.empty else 'N/A'}. "
            f"Returning default forecast for {n_assets_in_prices} assets."
        )
        return np.zeros(n_assets_in_prices), np.eye(n_assets_in_prices) * 1e-6

    # Calculate daily log returns: log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
    # .values gives numpy array. np.diff acts on rows (axis=0).
    log_prices = np.log(price_history_slice.values)
    daily_log_returns = np.diff(
        log_prices, axis=0
    )  # Results in (lookback_days-1) x n_assets matrix

    if not np.isfinite(daily_log_returns).all():
        log.warning(
            "calculate_mu_sigma_hat: Non-finite log returns calculated (e.g., from zero prices). Using default forecast."
        )
        # Count NaNs per asset for more detailed logging if needed
        nan_counts = np.sum(np.isnan(daily_log_returns), axis=0)
        if np.any(nan_counts > 0):
            log.debug(f"NaN counts in log returns per asset: {nan_counts}")
        return np.zeros(n_assets_in_prices), np.eye(n_assets_in_prices) * 1e-6

    mu_hat = np.mean(
        daily_log_returns, axis=0
    )  # 1D array of mean log returns per asset

    # Covariance calculation requires at least 2 data points (returns) for ddof=1
    if daily_log_returns.shape[0] < 2:
        log.warning(
            f"Only {daily_log_returns.shape[0]} returns available. "
            "Cannot calculate meaningful covariance. Using diagonal covariance."
        )
        # Use variance if at least 1 return, else zeros. Add small identity for invertibility.
        asset_variances = (
            np.var(daily_log_returns, axis=0, ddof=0)
            if daily_log_returns.shape[0] > 0
            else np.zeros(n_assets_in_prices)
        )
        sigma_hat = (
            np.diag(asset_variances) + np.eye(n_assets_in_prices) * 1e-6
        )  # Small value for stability
    else:
        # rowvar=False because each column is a variable (asset), each row is an observation (day)
        # However, np.cov default is rowvar=True (each row is a variable).
        # So, we need to transpose daily_log_returns if columns are assets.
        sigma_hat = np.cov(
            daily_log_returns, rowvar=False, ddof=1
        )  # Covariance matrix of returns
        # Add small diagonal perturbation for numerical stability if matrix is near singular
        sigma_hat = sigma_hat + np.eye(sigma_hat.shape[0]) * 1e-9

    return mu_hat, sigma_hat


def get_user_risk_budget_from_env(env: "FinancialRecEnv", user_id: str) -> float:
    """
    Retrieves the risk budget (sigma_cap) for a given user ID from the environment.
    Assumes env.user_configs_for_heuristics DataFrame exists with 'User' and 'σ_budget'.
    """
    if not hasattr(env, "user_configs_for_heuristics"):
        log.warning(
            "Environment instance does not have 'user_configs_for_heuristics'. Returning default risk budget 0.1."
        )
        return 0.1

    user_config_row = env.user_configs_for_heuristics[
        env.user_configs_for_heuristics["User"] == user_id
    ]
    if user_config_row.empty:
        log.warning(
            f"Risk budget not found for user_id: {user_id} in env.user_configs_for_heuristics. "
            f"Returning median budget or default 0.1."
        )
        if (
            not env.user_configs_for_heuristics.empty
            and "σ_budget" in env.user_configs_for_heuristics.columns
        ):
            return float(env.user_configs_for_heuristics["σ_budget"].median())
        return 0.1  # Default if no data or column missing

    return float(user_config_row.iloc[0]["σ_budget"])


def get_current_date_from_env(env: "FinancialRecEnv") -> pd.Timestamp:
    """
    Retrieves the current simulation date (start of the current month) from the environment.
    Assumes env.current_month_idx and env.time_index_monthly exist.
    """
    if not hasattr(env, "time_index_monthly") or not hasattr(env, "current_month_idx"):
        log.warning(
            "Environment instance missing 'time_index_monthly' or 'current_month_idx'. "
            "Returning current system time, which is likely incorrect for simulation."
        )
        return pd.Timestamp.now().normalize()

    if env.current_month_idx < 0 or env.current_month_idx >= len(
        env.time_index_monthly
    ):
        log.warning(
            f"Invalid current_month_idx: {env.current_month_idx} (max: {len(env.time_index_monthly) - 1}). "
            "Returning latest available simulation date or current system time."
        )
        return (
            env.time_index_monthly.max()
            if not env.time_index_monthly.empty
            else pd.Timestamp.now().normalize()
        )

    return env.time_index_monthly[env.current_month_idx]
