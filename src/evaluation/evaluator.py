# src/evaluation/evaluator.py
import numpy as np
import logging
from tqdm import tqdm
from typing import Callable, Any, Dict, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from environment.financial_env import FinancialRecEnv # Assuming flat src structure

log = logging.getLogger(__name__)

def evaluate_policy(
    policy_callable: Callable[..., Any], # Can return int or Tuple (action, ...)
    env: "FinancialRecEnv",
    num_episodes: int,
    policy_is_agent_instance: bool,
    policy_name: str,
    **kwargs_for_policy: Any,
) -> Tuple[float, float, float, List[float], List[int]]:
    """
    Evaluates a given policy (agent or heuristic) for a number of episodes.

    Args:
        policy_callable: The function/method to call to get an action.
                         - If agent: Expected to take `state` and potentially other kwargs
                           (like `epsilon`) passed via `kwargs_for_policy`.
                           Can return just `action_index` or a tuple `(action_index, ...)`.
                         - If heuristic/other: Expected to take `state` and context kwargs
                           (like `user_id`, `env_instance`, `prices_df`, etc.) which are
                           constructed here and merged with `kwargs_for_policy`.
        env: The environment instance. It will be reset for each episode.
        num_episodes: Number of episodes to run for evaluation.
        policy_is_agent_instance: True if policy_callable is an agent's method.
        policy_name: Name of the policy for logging and results.
        **kwargs_for_policy: Additional keyword arguments to pass to the policy_callable.
                             Examples:
                             - For DQNAgent: `epsilon=0.0`
                             - For A2CAgent: `store_for_trajectory=False` (if method supports it)
                             - For Random policy: `action_size=cfg.action_size`
                             - For Heuristics: `prices_df`, `cfg_heuristics`, `num_assets`

    Returns:
        Tuple: (avg_reward, std_reward, avg_length, all_rewards_list, all_lengths_list)
    """
    log.info(
        f"--- Starting Evaluation for Policy: {policy_name} ({num_episodes} episodes) ---"
    )
    all_rewards: List[float] = []
    all_lengths: List[int] = []

    for _episode_num in tqdm( # _episode_num not explicitly used inside loop other than for tqdm
        range(num_episodes), desc=f"Evaluating {policy_name}", leave=False
    ):
        state_numeric, info = env.reset()
        user_id = info.get("user_id") # For context if needed by heuristic

        cumulative_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False
        
        episode_specific_cache: Dict[str, Any] = {} # For heuristics if they use it

        while not (terminated or truncated):
            action_result: Any # Can be int or tuple

            if policy_is_agent_instance:
                # For agents, pass state and whatever is in kwargs_for_policy
                # (e.g., epsilon for DQN, store_for_trajectory for A2C during training)
                # For evaluation, A2C's choose_action might have store_for_trajectory=False by default
                # or we can pass it via kwargs_for_policy from run.py.
                action_result = policy_callable(state_numeric, **kwargs_for_policy)
            else:
                # For non-agent policies (Random, Heuristics)
                # Construct context and merge with provided kwargs
                current_call_kwargs = {
                    "user_id": user_id,
                    "env_instance": env,
                    "episode_cache": episode_specific_cache,
                    **kwargs_for_policy, # Contains action_size, prices_df, cfg_heuristics etc.
                }
                action_result = policy_callable(state_numeric, **current_call_kwargs)

            # Extract action_index if policy_callable returns a tuple (e.g., A2C returns action, log_prob, value)
            if isinstance(action_result, tuple):
                action_index = int(action_result[0]) # Assume action is the first element
            else:
                action_index = int(action_result)

            if not (0 <= action_index < env.num_actions):
                log.error(f"Policy '{policy_name}' returned invalid action_index: {action_index}. Env num_actions: {env.num_actions}. Forcing action 0.")
                action_index = 0 # Fallback to a valid action to prevent crash

            next_state_numeric, reward, terminated, truncated, _info = env.step(action_index)

            state_numeric = next_state_numeric
            cumulative_reward += reward
            step_count += 1

        all_rewards.append(cumulative_reward)
        all_lengths.append(step_count)

    avg_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
    std_reward = float(np.std(all_rewards)) if all_rewards else 0.0
    avg_length = float(np.mean(all_lengths)) if all_lengths else 0.0

    log.info(f"--- Evaluation Finished for Policy: {policy_name} ---")
    log.info(f"  Avg Reward: {avg_reward:.2f} +/- {std_reward:.2f}")
    log.info(f"  Avg Ep Length: {avg_length:.1f} steps")
    return avg_reward, std_reward, avg_length, all_rewards, all_lengths