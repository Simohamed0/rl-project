# src/utils/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any
from omegaconf import DictConfig  # For config_for_titles type hint

# Import PLOTLY_OK if Optuna plots are added here later
# from .common_utils import PLOTLY_OK
# if PLOTLY_OK:
#     try:
#         import plotly.graph_objects as go
#         from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
#     except ImportError:
#         pass # PLOTLY_OK would be false if initial import failed

log = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")  # Using a seaborn style
plt.rcParams.update(
    {"font.size": 10, "figure.facecolor": "white", "savefig.facecolor": "white"}
)


def plot_training_results(
    results_dict: Dict[str, List[Any]],
    config_for_titles: DictConfig,  # Pass relevant part of cfg for titles
    save_dir: str,
    policy_name_prefix: str = "DQN",
) -> None:
    """
    Plots training rewards, episode lengths, loss, and epsilon decay.

    Args:
        results_dict: Dictionary containing lists for 'rewards', 'lengths',
                      'losses' (optional), 'epsilon' (optional).
        config_for_titles: Config object (e.g., cfg.training or main cfg)
                           to extract info like num_episodes for titles.
        save_dir: Directory to save the plots.
        policy_name_prefix: Prefix for plot titles and filenames (e.g., "DQN_Final").
    """
    os.makedirs(save_dir, exist_ok=True)
    log.info(
        f"Generating training plots for {policy_name_prefix}, saving to {save_dir}"
    )

    num_total_episodes = config_for_titles.training.get(
        "num_episodes", len(results_dict.get("rewards", []))
    )
    window_size = max(
        1, min(50, num_total_episodes // 20 if num_total_episodes > 0 else 1)
    )

    # 1. Plot Rewards
    if "rewards" in results_dict and results_dict["rewards"]:
        try:
            plt.figure(figsize=(10, 6))
            rewards = results_dict["rewards"]
            plt.plot(
                rewards,
                label=f"{policy_name_prefix} Raw Reward",
                alpha=0.4,
                linewidth=1,
                color="tab:green",
            )
            if len(rewards) >= window_size:
                rolling_rewards = (
                    pd.Series(rewards).rolling(window=window_size, min_periods=1).mean()
                )
                plt.plot(
                    rolling_rewards,
                    label=f"Rolling Avg (w={window_size})",
                    color="green",
                    linewidth=2,
                )

            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title(
                f"{policy_name_prefix} Training Rewards (Total Episodes: {num_total_episodes})"
            )
            plt.legend(loc="best")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plot_path = os.path.join(
                save_dir, f"{policy_name_prefix.lower()}_training_rewards.png"
            )
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log.debug(f"Saved training rewards plot to {plot_path}")
        except Exception as e:
            log.error(
                f"Error generating {policy_name_prefix} rewards plot: {e}",
                exc_info=True,
            )
    else:
        log.info(f"No reward data to plot for {policy_name_prefix}.")

    # 2. Plot Episode Lengths
    if "lengths" in results_dict and results_dict["lengths"]:
        try:
            plt.figure(figsize=(10, 6))
            lengths = results_dict["lengths"]
            plt.plot(
                lengths,
                label=f"{policy_name_prefix} Raw Length",
                alpha=0.4,
                linewidth=1,
                color="tab:blue",
            )
            if len(lengths) >= window_size:
                rolling_lengths = (
                    pd.Series(lengths).rolling(window=window_size, min_periods=1).mean()
                )
                plt.plot(
                    rolling_lengths,
                    label=f"Rolling Avg (w={window_size})",
                    color="blue",
                    linewidth=2,
                )

            plt.xlabel("Episode")
            plt.ylabel("Episode Length (Steps)")
            plt.title(f"{policy_name_prefix} Training Episode Lengths")
            plt.legend(loc="best")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plot_path = os.path.join(
                save_dir, f"{policy_name_prefix.lower()}_training_lengths.png"
            )
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log.debug(f"Saved training lengths plot to {plot_path}")
        except Exception as e:
            log.error(
                f"Error generating {policy_name_prefix} lengths plot: {e}",
                exc_info=True,
            )
    else:
        log.info(f"No episode length data to plot for {policy_name_prefix}.")

    # 3. Plot Training Loss (if available)
    if (
        "losses" in results_dict
        and results_dict["losses"]
        and any(results_dict["losses"])
    ):  # Check if list is not all None/0
        try:
            plt.figure(figsize=(10, 6))
            losses = [
                l for l in results_dict["losses"] if l is not None
            ]  # Filter out Nones if any
            if not losses:  # If all were None
                raise ValueError("Loss data contains only None values.")

            plt.plot(
                losses,
                label="Avg Loss per Episode",
                color="tab:purple",
                linewidth=1,
                alpha=0.5,
            )
            if len(losses) >= window_size:
                rolling_loss = (
                    pd.Series(losses).rolling(window=window_size, min_periods=1).mean()
                )
                plt.plot(
                    rolling_loss,
                    label=f"Rolling Avg (w={window_size})",
                    color="purple",
                    linewidth=2,
                )

            plt.xlabel("Episode (where learning occurred)")
            plt.ylabel("Average Loss (e.g., MSE)")
            plt.title(f"{policy_name_prefix} Training Loss")
            plt.legend(loc="best")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plot_path = os.path.join(
                save_dir, f"{policy_name_prefix.lower()}_training_loss.png"
            )
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log.debug(f"Saved training loss plot to {plot_path}")
        except Exception as e:
            log.error(
                f"Error generating {policy_name_prefix} loss plot: {e}", exc_info=True
            )
    else:
        log.info(
            f"No loss data (or insufficient data) to plot for {policy_name_prefix}."
        )

    # 4. Plot Epsilon Decay (if available)
    if "epsilon" in results_dict and results_dict["epsilon"]:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(
                results_dict["epsilon"],
                label="Epsilon Value",
                color="tab:red",
                linewidth=1.5,
            )
            plt.xlabel("Episode")
            plt.ylabel("Epsilon")
            plt.title(f"{policy_name_prefix} Epsilon Decay Schedule")
            plt.legend(loc="upper right")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plot_path = os.path.join(
                save_dir, f"{policy_name_prefix.lower()}_epsilon_decay.png"
            )
            plt.savefig(plot_path, dpi=150)
            plt.close()
            log.debug(f"Saved epsilon decay plot to {plot_path}")
        except Exception as e:
            log.error(
                f"Error generating {policy_name_prefix} epsilon plot: {e}",
                exc_info=True,
            )
    else:
        log.info(f"No epsilon data to plot for {policy_name_prefix}.")


def plot_evaluation_comparison(
    eval_results_dict: Dict[str, Dict[str, Any]],
    num_eval_episodes_for_title: int,
    save_dir: str,
) -> None:
    """
    Plots a bar chart comparing average evaluation rewards of different policies.

    Args:
        eval_results_dict: Dictionary where keys are policy names and values are
                           dictionaries containing 'avg_reward', 'std_reward', 'rewards'.
        num_eval_episodes_for_title: Number of evaluation episodes (for plot title).
        save_dir: Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Generating evaluation comparison plot, saving to {save_dir}")

    policy_names = list(eval_results_dict.keys())

    # Filter out policies with no valid reward data (e.g. if evaluation failed or was skipped)
    valid_policies_data = {
        name: data
        for name, data in eval_results_dict.items()
        if data
        and "avg_reward" in data
        and data.get("rewards")  # Ensure rewards list exists for std dev
    }

    if not valid_policies_data:
        log.warning("No valid evaluation results found to plot comparison.")
        return

    sorted_policy_names = sorted(
        valid_policies_data.keys(),
        key=lambda name: valid_policies_data[name]["avg_reward"],
        reverse=True,
    )

    avg_rewards = [
        valid_policies_data[name]["avg_reward"] for name in sorted_policy_names
    ]
    # Use pre-calculated std_reward if available, otherwise calculate from 'rewards' list
    std_rewards = [
        valid_policies_data[name].get(
            "std_reward",
            np.std(valid_policies_data[name].get("rewards", [0]))
            if valid_policies_data[name].get("rewards")
            else 0,
        )
        for name in sorted_policy_names
    ]

    try:
        plt.figure(
            figsize=(max(8, len(sorted_policy_names) * 1.5), 6)
        )  # Adjust width based on num policies

        colors = []
        palette = sns.color_palette("viridis", len(sorted_policy_names))
        for i, name in enumerate(sorted_policy_names):
            if "DQN" in name:
                colors.append("darkgreen")
            elif "Random" in name:
                colors.append("grey")
            else:
                colors.append(
                    palette[i % len(palette)]
                )  # Cycle through palette for heuristics

        bars = plt.bar(
            sorted_policy_names,
            avg_rewards,
            yerr=std_rewards,
            capsize=5,
            color=colors,
            alpha=0.8,
        )

        plt.ylabel("Average Cumulative Reward")
        plt.title(
            f"Policy Performance Comparison ({num_eval_episodes_for_title} Eval Episodes)"
        )
        plt.xticks(rotation=30, ha="right")  # Rotate labels if many policies
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Add text labels on bars
        for bar in bars:
            yval = bar.get_height()
            y_offset = (
                max(avg_rewards) - min(min(avg_rewards), 0)
            ) * 0.05  # 5% of range as offset
            text_y = yval + np.sign(yval) * y_offset if yval != 0 else y_offset
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                text_y,
                f"{yval:.2f}",
                ha="center",
                va="bottom" if yval >= 0 else "top",
                fontsize=9,
            )

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "policy_evaluation_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log.debug(f"Saved evaluation comparison plot to {plot_path}")
    except Exception as e:
        log.error(f"Error generating evaluation comparison plot: {e}", exc_info=True)


# Add Optuna plotting functions here later if needed, guarded by PLOTLY_OK
# def plot_optuna_visualizations(study, save_dir): ...
