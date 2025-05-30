# src/heuristics/policies.py
import numpy as np
import pandas as pd
import logging
from typing import Callable, Dict, Any, TYPE_CHECKING, Optional
from omegaconf import DictConfig
import random  # For fallback if CVXPY MVO fails or for dummy

# Import utilities from the same package
from .utils import (
    normalise_weights,
    calculate_mu_sigma_hat,
    get_user_risk_budget_from_env,
    get_current_date_from_env,
)

# Import global CVXPY_OK check
from utils.common_utils import CVXPY_OK  # Assumes common_utils is accessible

if CVXPY_OK:
    import cvxpy as cp  # type: ignore

if TYPE_CHECKING:
    from environment.financial_env import FinancialRecEnv  # Assuming flat src structure

log = logging.getLogger(__name__)

# --- Weight Calculation Functions ---


def ew_weights(num_assets: int) -> np.ndarray:
    """Returns equal weights for a given number of assets."""
    if num_assets <= 0:
        log.warning("ew_weights: num_assets is non-positive. Returning empty array.")
        return np.array([])
    return np.full(num_assets, 1.0 / num_assets)


def rp_weights(sigma_hat: np.ndarray) -> np.ndarray:
    """
    Calculates Risk Parity weights (standard version: inversely proportional to volatility).
    Args:
        sigma_hat: Covariance matrix of asset returns.
    """
    if (
        sigma_hat.ndim != 2
        or sigma_hat.shape[0] != sigma_hat.shape[1]
        or sigma_hat.shape[0] == 0
    ):
        log.warning(
            f"rp_weights: Invalid sigma_hat shape {sigma_hat.shape}. Returning empty array or EW if possible."
        )
        return (
            ew_weights(sigma_hat.shape[0]) if sigma_hat.shape[0] > 0 else np.array([])
        )

    asset_variances = np.diag(sigma_hat)
    # Handle potential negative variances if sigma_hat is not perfectly PSD (e.g. from estimation)
    asset_volatilities = np.sqrt(
        np.maximum(asset_variances, 1e-12)
    )  # Floor variance at a tiny positive value

    # Inverse volatilities, handle division by zero for zero-volatility assets
    inv_volatilities = np.zeros_like(asset_volatilities)
    non_zero_vol_mask = asset_volatilities > 1e-9  # Small epsilon
    inv_volatilities[non_zero_vol_mask] = 1.0 / asset_volatilities[non_zero_vol_mask]

    sum_inv_vol = np.sum(inv_volatilities)
    if sum_inv_vol < 1e-9:  # All assets have near-zero volatility or all were zero
        log.warning(
            "rp_weights: All assets have near-zero volatility or sum of inverse volatilities is zero. Falling back to Equal Weight."
        )
        return ew_weights(sigma_hat.shape[0])

    raw_rp_weights = inv_volatilities / sum_inv_vol
    return normalise_weights(raw_rp_weights)  # Final normalization for safety


def mvo_weights(
    mu_hat: np.ndarray,
    sigma_hat: np.ndarray,
    sigma_cap: float,
    num_assets: int,  # For fallback
    cvxpy_solver: Optional[str] = None,
) -> np.ndarray:
    """
    Calculates Mean-Variance Optimized weights.
    Maximizes expected return subject to a risk cap (volatility constraint).

    Args:
        mu_hat: Expected daily returns.
        sigma_hat: Covariance matrix of daily returns.
        sigma_cap: User's risk budget (target volatility, daily if mu/sigma are daily).
        num_assets: Total number of assets (for EW fallback).
        cvxpy_solver: Specific solver for CVXPY (e.g., "SCS", "ECOS"). None for default.
    """
    if not CVXPY_OK:
        log.info("mvo_weights: CVXPY not available. Falling back to Equal Weight.")
        return ew_weights(num_assets)

    if mu_hat.shape[0] != num_assets or sigma_hat.shape != (num_assets, num_assets):
        log.warning(
            f"mvo_weights: Mismatch in dimensions of mu_hat ({mu_hat.shape}), sigma_hat ({sigma_hat.shape}), "
            f"and num_assets ({num_assets}). Falling back to EW."
        )
        return ew_weights(num_assets)
    if num_assets == 0:
        return np.array([])

    w = cp.Variable(num_assets)
    objective = cp.Maximize(mu_hat @ w)
    # Risk constraint: portfolio variance <= sigma_cap^2
    # Ensure sigma_cap is positive
    risk_cap_squared = (
        max(sigma_cap, 1e-6) ** 2
    )  # Use a small floor for sigma_cap to avoid issues with zero

    constraints = [
        cp.sum(w) == 1,  # Weights sum to 1
        w >= 0,  # No short selling
        cp.quad_form(w, sigma_hat) <= risk_cap_squared,  # Portfolio variance constraint
    ]
    problem = cp.Problem(objective, constraints)

    try:
        # Solve the problem
        # verbose=False is good for non-interactive runs
        # Specify solver if provided and available, otherwise let CVXPY choose
        solver_arg = (
            cvxpy_solver
            if cvxpy_solver and cvxpy_solver in cp.installed_solvers()
            else None
        )
        if cvxpy_solver and solver_arg is None:
            log.warning(
                f"CVXPY solver '{cvxpy_solver}' not installed. Letting CVXPY choose a default solver."
            )

        problem.solve(solver=solver_arg, verbose=False)

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if w.value is not None:
                return normalise_weights(w.value)  # Normalize for safety and positivity
            else:
                log.warning(
                    f"mvo_weights: CVXPY solve status is {problem.status} but weights are None. Falling back to EW."
                )
        else:
            log.warning(
                f"mvo_weights: CVXPY optimization failed or problem infeasible/unbounded. Status: {problem.status}. Falling back to EW."
            )
    except Exception as e:
        log.error(
            f"mvo_weights: Error during CVXPY optimization: {e}. Falling back to EW.",
            exc_info=True,
        )

    return ew_weights(num_assets)


# --- Main Heuristic Policy Decision Function Wrapper ---


def get_heuristic_policy_decision_func(
    strategy_name: str,
    # These are needed for the returned callable to function within evaluate_policy
    # They are passed via **kwargs_for_policy from run.py's call to evaluate_policy
    # prices_df: pd.DataFrame, -> This will be pivoted inside the callable
    # cfg_heuristics: DictConfig,
    # num_assets: int
) -> Callable[[np.ndarray, Dict[str, Any]], int]:
    """
    Returns a policy decision function for the specified heuristic strategy.
    The returned function is compatible with `evaluate_policy`'s `policy_callable`.

    It handles caching of mu_hat, sigma_hat, and target_weights per episode (per user).
    """
    log.info(
        f"Creating heuristic policy decision function for strategy: {strategy_name}"
    )

    def policy_decision_function(
        state_numeric: np.ndarray,  # Current state from env (not directly used by these heuristics)
        # Expected kwargs passed from evaluate_policy's `current_call_kwargs`
        user_id: str,
        env_instance: "FinancialRecEnv",  # For getting current date and user risk budget
        episode_cache: Dict[str, Any],  # For caching weights/forecasts per episode
        prices_df: pd.DataFrame,  # Long format: ISIN, timestamp, closePrice
        cfg_heuristics: DictConfig,
        num_assets: int,
        **ignored_kwargs,  # To absorb any other kwargs from evaluate_policy
    ) -> int:
        """
        Inner function that makes the decision for a single step.
        Caches forecast (mu, Sigma) and target weights for the current user for the episode.
        """
        cache_key_forecast = f"{user_id}_forecast"
        cache_key_weights = f"{user_id}_weights_{strategy_name}"

        if (
            cache_key_weights not in episode_cache
        ):  # Calculate weights only once per episode for this user/strategy
            if cache_key_forecast not in episode_cache:
                current_sim_date = get_current_date_from_env(env_instance)
                lookback = cfg_heuristics.lookback_days

                # Pivot prices_df (long) to wide format for calculate_mu_sigma_hat
                # Ensure ISINs in prices_df match those expected by the environment's action space
                # This assumes `env_instance.asset_isins_list` contains the relevant ISINs
                # and `prices_df` contains data for these.
                relevant_isins = env_instance.asset_isins_list
                prices_df_filtered = prices_df[prices_df["ISIN"].isin(relevant_isins)]
                if prices_df_filtered.empty:
                    log.warning(
                        f"No price data for relevant assets for user {user_id}, date {current_sim_date}. Using default forecast."
                    )
                    mu_hat_val = np.zeros(num_assets)
                    sigma_hat_val = np.eye(num_assets) * 1e-6
                else:
                    prices_df_wide_fmt = prices_df_filtered.pivot(
                        index="timestamp", columns="ISIN", values="closePrice"
                    )
                    # Reindex to ensure all relevant assets are present, fill NaNs (e.g., forward fill)
                    prices_df_wide_fmt = (
                        prices_df_wide_fmt.reindex(columns=relevant_isins)
                        .ffill()
                        .bfill()
                    )

                    mu_hat_val, sigma_hat_val = calculate_mu_sigma_hat(
                        prices_df_wide_fmt, current_sim_date, lookback, num_assets
                    )
                episode_cache[cache_key_forecast] = (mu_hat_val, sigma_hat_val)
            else:
                mu_hat_val, sigma_hat_val = episode_cache[cache_key_forecast]

            sigma_cap_val = get_user_risk_budget_from_env(env_instance, user_id)

            target_weights: np.ndarray
            if strategy_name == "MVO":
                cvxpy_solver_from_cfg = cfg_heuristics.get("cvxpy_solver", None)
                target_weights = mvo_weights(
                    mu_hat_val,
                    sigma_hat_val,
                    sigma_cap_val,
                    num_assets,
                    cvxpy_solver_from_cfg,
                )
            elif strategy_name == "Risk Parity":
                target_weights = rp_weights(sigma_hat_val)
            elif strategy_name == "Equal Weight":
                target_weights = ew_weights(num_assets)
            else:
                log.error(
                    f"Unknown heuristic strategy: {strategy_name}. Defaulting to random action."
                )
                return random.choice(np.arange(num_assets)) if num_assets > 0 else 0

            episode_cache[cache_key_weights] = target_weights
        else:
            target_weights = episode_cache[cache_key_weights]

        if (
            target_weights.size == 0 or num_assets == 0
        ):  # Should not happen if num_assets > 0
            log.warning(
                f"Heuristic {strategy_name} produced empty target_weights or num_assets is 0. Choosing random action."
            )
            return random.choice(np.arange(num_assets)) if num_assets > 0 else 0

        # Choose action: index of the asset with the highest target weight.
        # This assumes the order of assets in mu_hat/sigma_hat (from prices_df_wide_fmt.columns)
        # matches the environment's internal action indexing (env_instance.asset_isins_list).
        # calculate_mu_sigma_hat uses prices_df_wide.shape[1] which should align if prices_df_wide
        # columns are ordered like env_instance.asset_isins_list (due to reindex).
        action_index = int(np.argmax(target_weights))

        if not (0 <= action_index < num_assets):
            log.warning(
                f"Heuristic {strategy_name} generated invalid action_index {action_index}. Falling back to random."
            )
            action_index = random.choice(np.arange(num_assets)) if num_assets > 0 else 0

        return action_index

    return policy_decision_function
