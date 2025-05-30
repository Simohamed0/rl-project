# src/run.py
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import logging
import os
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm
import random # For Random Policy evaluation
# import json # Not strictly needed now without Optuna param saving

# --- Import project modules ---
from utils.common_utils import (
    set_seeds,
    get_torch_device,
    setup_data_and_environment,
    CVXPY_OK,
    # PLOTLY_OK, # Not used in this version
)
from agent.dqn_agent import DQNAgent
from agent.a2c_agent import A2CAgent # Import A2C Agent
from training.core_train_loops import train_dqn_episode, train_a2c_episode # Import both episode trainers
from evaluation.evaluator import evaluate_policy

# --- Dummy/Placeholder Plotting and Heuristic Functions (as before) ---
log_plot = logging.getLogger("plotting_placeholders")
log_heuristic = logging.getLogger("heuristic_placeholders")

def plot_training_results(results_dict, config_for_titles, save_dir, policy_name_prefix):
    log_plot.info(f"PLOTTING_PLACEHOLDER: Plotting training for {policy_name_prefix} in {save_dir}")
    # Example: with open(os.path.join(save_dir, f"{policy_name_prefix}_training_dummy.txt"), "w") as f: f.write("dummy")

def plot_evaluation_comparison(eval_results_dict, num_eval_episodes_for_title, save_dir):
    log_plot.info(f"PLOTTING_PLACEHOLDER: Plotting eval comparison in {save_dir}")

def get_heuristic_policy_decision_func(strategy_name: str, **kwargs):
    log_heuristic.warning(f"HEURISTIC_PLACEHOLDER for {strategy_name}. Using random policy.")
    action_size = kwargs.get("num_assets", 1)
    if action_size == 0: action_size = 1
    return lambda state, **current_call_kwargs: random.choice(np.arange(action_size))

log = logging.getLogger(__name__) # Main logger for run.py


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main_app(cfg: DictConfig) -> None:
    try:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except Exception:
        log.warning("Could not get output_dir via HydraConfig. Using os.getcwd().")
        output_dir = os.getcwd()

    log.info(f"Effective Hydra output directory: {output_dir}")
    log.info(f"Run Config: Seed={cfg.seed}, Device={cfg.device}, UseRealData={cfg.use_real_data}, AgentType={cfg.agent.type}")
    log.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # --- 1. Initial Setup ---
    set_seeds(cfg.seed)
    device = get_torch_device(cfg.device)
    with open_dict(cfg): cfg.device_actual = str(device)

    # --- 2. Load Data and Setup Environment ---
    try:
        env, env_data_dict, _ = setup_data_and_environment(cfg)
        prices_df_for_eval = env_data_dict["prices"]
    except Exception as e:
        log.error(f"Failed during data/env setup: {e}", exc_info=True); sys.exit(1)

    all_eval_results = {}

    # --- Evaluate Baselines (Random & Heuristics) ---
    #if cfg.evaluate_baselines:
    #    log.info("\n--- Evaluating Random Policy ---")
    #    def random_policy_callable(state, action_size, **ignored_kwargs):
    #        return random.choice(np.arange(action_size))
    #    rand_avg_r, rand_std_r, rand_avg_l, rand_all_rewards, _ = evaluate_policy(
    #        random_policy_callable, env, cfg.evaluation.num_random_episodes,
    #        False, "Random", action_size=cfg.action_size)
    #    all_eval_results["Random"] = {"avg_reward": rand_avg_r, "std_reward": rand_std_r, "avg_length": rand_avg_l, "rewards": rand_all_rewards}
    #    log.info(f"Random Policy: Avg Reward={rand_avg_r:.2f} +/- {rand_std_r:.2f}")
#
    #    log.info("\n--- Evaluating Heuristic Policies (Dummy) ---")
    #    for strategy_name in ["MVO", "Risk Parity", "Equal Weight"]:
    #        if strategy_name == "MVO" and not CVXPY_OK: log.warning("Skip MVO: CVXPY missing."); continue
    #        heuristic_callable = get_heuristic_policy_decision_func(strategy_name, num_assets=cfg.num_assets)
    #        h_avg_r, h_std_r, h_avg_l, h_all_rewards, _ = evaluate_policy(
    #            heuristic_callable, env, cfg.evaluation.num_eval_episodes,
    #            False, strategy_name, prices_df=prices_df_for_eval,
    #            cfg_heuristics=cfg.heuristics, num_assets=cfg.num_assets)
    #        all_eval_results[strategy_name] = {"avg_reward": h_avg_r, "std_reward": h_std_r, "avg_length": h_avg_l, "rewards": h_all_rewards}
    #        log.info(f"{strategy_name} (Dummy): Avg Reward={h_avg_r:.2f} +/- {h_std_r:.2f}")
    #else:
    #    log.info("Skipping baseline evaluations as per configuration.")


    # --- 3. Optuna (Placeholder) ---
    best_hyperparams_from_optuna = {}
    # study_obj_for_plotting = None # For Optuna plots later
    if cfg.run_optuna: log.warning("Optuna block is a placeholder. Not running Optuna.")
    else: log.info("Optuna optimization disabled by config.")


    # --- 4. Initialize and Train Agent ---
    agent_instance = None # Can be DQNAgent or A2CAgent
    training_results = {} # To store training metrics

    if cfg.train_final_agent:
        log.info(f"\n--- Initializing Agent (Type: {cfg.agent.type.upper()}) ---")
        agent_cfg_resolved = OmegaConf.to_container(cfg.agent, resolve=True) # Get mutable dict
        if best_hyperparams_from_optuna: # If Optuna ran and we have results
            log.info(f"Applying Optuna best params: {best_hyperparams_from_optuna}")
            # Add merging logic here if Optuna param names/structure differs
            agent_cfg_resolved.update(best_hyperparams_from_optuna)
        
        agent_hydra_cfg_final = OmegaConf.create(agent_cfg_resolved) # Back to DictConfig if agent expects it

        try:
            if cfg.agent.type.upper() == "DQN":
                agent_instance = DQNAgent(
                    state_size=cfg.state_size, action_size=cfg.action_size,
                    agent_cfg=agent_hydra_cfg_final, global_seed=cfg.seed, device=device)
            elif cfg.agent.type.upper() == "A2C":
                agent_instance = A2CAgent(
                    state_size=cfg.state_size, action_size=cfg.action_size,
                    agent_cfg=agent_hydra_cfg_final, global_seed=cfg.seed, device=device)
            else:
                log.error(f"Unknown agent type: {cfg.agent.type}"); sys.exit(1)
        except Exception as e:
            log.error(f"Failed to initialize {cfg.agent.type} agent: {e}", exc_info=True); sys.exit(1)

        log.info(f"--- Starting {cfg.agent.type.upper()} Training for {cfg.training.num_episodes} episodes ---")
        rewards_log, lengths_log, losses1_log, losses2_log, eps_log = [], [], [], [], []
        
        if cfg.agent.type.upper() == "DQN": epsilon = cfg.training.epsilon_start

        for ep_num in tqdm(range(1, cfg.training.num_episodes + 1), desc=f"{cfg.agent.type.upper()} Training"):
            if cfg.agent.type.upper() == "DQN":
                ep_r, ep_s, ep_l1 = train_dqn_episode(env, agent_instance, epsilon) # ep_l1 is dqn_loss
                losses1_log.append(ep_l1 if ep_l1 is not None else np.nan)
                epsilon = max(cfg.training.epsilon_end, epsilon * cfg.training.epsilon_decay)
                eps_log.append(epsilon)
                loss_str = f"DQN_Loss: {ep_l1 if ep_l1 is not None else float('nan'):.4f}"
            elif cfg.agent.type.upper() == "A2C":
                ep_r, ep_s, ep_l1, ep_l2 = train_a2c_episode(env, agent_instance) # ep_l1 actor, ep_l2 critic
                losses1_log.append(ep_l1 if ep_l1 is not None else np.nan)
                losses2_log.append(ep_l2 if ep_l2 is not None else np.nan)
                loss_str = f"ActorL: {ep_l1 if ep_l1 is not None else float('nan'):.4f}, CriticL: {ep_l2 if ep_l2 is not None else float('nan'):.4f}"
            
            rewards_log.append(ep_r); lengths_log.append(ep_s)

            if ep_num % 100 == 0 or ep_num == cfg.training.num_episodes:
                avg_r_100 = np.mean(rewards_log[-100:]) if rewards_log else 0.0
                eps_str = f"Eps: {epsilon:.4f}" if cfg.agent.type.upper() == "DQN" else "PolicyStochastic"
                log.info(f"Ep {ep_num}/{cfg.training.num_episodes} | AvgR(100): {avg_r_100:.2f} | {eps_str} | {loss_str}")

        log.info(f"--- {cfg.agent.type.upper()} Training Finished ---")
        
        # Standardized save path prefix. Agent's save_model method handles specific filenames.
        model_save_prefix = os.path.join(output_dir, f"{cfg.agent.type.lower()}_agent_final")
        if hasattr(agent_instance, 'save_model'):
            agent_instance.save_model(model_save_prefix)
        else:
            log.warning(f"Agent {cfg.agent.type} does not have a save_model method.")


        training_results = {"rewards": rewards_log, "lengths": lengths_log}
        if cfg.agent.type.upper() == "DQN":
            training_results["losses"] = losses1_log
            training_results["epsilon"] = eps_log
        elif cfg.agent.type.upper() == "A2C":
            training_results["actor_losses"] = losses1_log
            training_results["critic_losses"] = losses2_log
        
        summary_df = pd.DataFrame(training_results)
        summary_path = os.path.join(output_dir, f"{cfg.agent.type.lower()}_training_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        log.info(f"{cfg.agent.type.upper()} training summary saved to {summary_path}")
    else:
        log.info("Skipping agent training as per 'train_final_agent: false'.")
        # Logic to load pre-trained agent if evaluation is requested
        if cfg.evaluate_final_agent:
            load_path_prefix = os.path.join(output_dir, f"{cfg.agent.type.lower()}_agent_final") # Standardized
            # Could also check cfg.agent_load_path if you add that to config
            
            log.info(f"Attempting to load pre-trained {cfg.agent.type.upper()} agent from prefix: {load_path_prefix}")
            # Re-initialize an agent instance to load into
            agent_cfg_load = OmegaConf.to_container(cfg.agent, resolve=True)
            agent_hydra_cfg_load = OmegaConf.create(agent_cfg_load)
            if cfg.agent.type.upper() == "DQN":
                agent_instance = DQNAgent(state_size=cfg.state_size, action_size=cfg.action_size,
                                          agent_cfg=agent_hydra_cfg_load, global_seed=cfg.seed, device=device)
            elif cfg.agent.type.upper() == "A2C":
                 agent_instance = A2CAgent(state_size=cfg.state_size, action_size=cfg.action_size,
                                          agent_cfg=agent_hydra_cfg_load, global_seed=cfg.seed, device=device)
            
            if agent_instance and hasattr(agent_instance, 'load_model') and agent_instance.load_model(load_path_prefix):
                log.info(f"Successfully loaded pre-trained {cfg.agent.type.upper()} agent.")
            else:
                log.warning(f"Could not load pre-trained {cfg.agent.type.upper()} agent. Evaluation of final agent might fail or use uninitialized agent.")
                agent_instance = None # Ensure it's None if load failed


    # --- 5. Evaluate Trained Agent ---
    if cfg.evaluate_final_agent and agent_instance is not None:
        log.info(f"\n--- Evaluating Trained {cfg.agent.type.upper()} Agent ---")
        eval_env, _, _ = setup_data_and_environment(cfg)
        
        agent_callable_for_eval = agent_instance.choose_action
        eval_specific_kwargs = {} # Start with empty kwargs
    
        if cfg.agent.type.upper() == "DQN":
            eval_specific_kwargs["epsilon"] = 0.0 # Greedy for DQN
        elif cfg.agent.type.upper() == "A2C":
            # For A2C evaluation:
            # - We usually don't want to store experiences for training.
            # - The current A2C choose_action samples from the policy.
            #   If you wanted deterministic A2C eval (argmax of policy),
            #   choose_action would need a flag or a separate eval_choose_action method.
            # Assuming we want stochastic A2C eval and want to ensure no trajectory storage:
            if hasattr(agent_instance, 'choose_action') and \
               'store_for_trajectory' in agent_instance.choose_action.__code__.co_varnames:
                 eval_specific_kwargs["store_for_trajectory"] = False
            # No epsilon for A2C
        
        agent_avg_r, agent_std_r, agent_avg_l, agent_all_rewards, _ = evaluate_policy(
            policy_callable=agent_callable_for_eval,
            env=eval_env,
            num_episodes=cfg.evaluation.num_eval_episodes,
            policy_is_agent_instance=True,
            policy_name=f"{cfg.agent.type.upper()} (Trained)",
            **eval_specific_kwargs # Pass the constructed kwargs
        )

        
        all_eval_results[f"{cfg.agent.type.upper()} (Trained)"] = {
            "avg_reward": agent_avg_r, "std_reward": agent_std_r,
            "avg_length": agent_avg_l, "rewards": agent_all_rewards
        }
        log.info(f"{cfg.agent.type.upper()} (Trained): Avg Reward={agent_avg_r:.2f} +/- {agent_std_r:.2f}")
    elif cfg.evaluate_final_agent:
        log.warning("Evaluation of final agent requested, but agent is not available (not trained or loaded).")


    # --- 6. Generate Plots (Using Placeholders) ---
    log.info("\n--- Generating Plots (Using Placeholders) ---")
    if training_results: # If training happened
        plot_training_results(training_results, cfg, output_dir, f"{cfg.agent.type.upper()}_Final")
    if all_eval_results: # If any evaluation happened
        plot_evaluation_comparison(all_eval_results, cfg.evaluation.num_eval_episodes, output_dir)


    # --- 7. Print Summary ---
    log.info("\n--- Run Summary ---")
    log.info(f"Agent Type: {cfg.agent.type.upper()}")
    # ... (other summary printing as before) ...
    if all_eval_results:
        log.info(f"\nPolicy Evaluation Results ({cfg.evaluation.num_eval_episodes} episodes each):")
        # ... (table printing logic from your updated run.py) ...
        header = f"{'Policy':<25} | {'Avg Reward':<12} | {'Std Reward':<12} | {'Avg Length':<12}"
        log.info(header); log.info("-" * len(header))
        sorted_policies = sorted(all_eval_results.items(), key=lambda item: item[1]["avg_reward"], reverse=True)
        for name, res in sorted_policies:
            log.info(f"{name:<25} | {res['avg_reward']:<12.2f} | {res.get('std_reward', float('nan')):<12.2f} | {res['avg_length']:<12.1f}")
        log.info("-" * len(header))

    log.info(f"\nRun finished. Outputs in: {output_dir}")
    log.info("--- Financial RL Project Script Completed ---")

if __name__ == "__main__":
    main_app()