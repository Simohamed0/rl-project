# Financial Asset Recommendation RL Agent

## Project Goal

The primary goal of this project is to develop a Reinforcement Learning (RL) agent capable of providing personalized financial asset recommendations to customers. The agent learns an optimal policy by interacting with a simulated financial environment. The key objectives for the agent's recommendations are:

1.  **Risk Alignment:** Suggest assets suitable for the customer's stated risk tolerance.
2.  **Portfolio Diversification:** Encourage the construction of well-diversified portfolios to mitigate unsystematic risk.
3.  **Performance Consideration:** Favor assets with positive recent performance or outlook.
4.  **Outperform Baselines:** Achieve better performance metrics compared to random recommendations and standard heuristic portfolio allocation strategies.

## Core Components

The project is structured around several key components:

### 1. Simulated Financial Environment (`FinancialRecEnv`)

*   **Purpose:** Provides a playground for the RL agent to learn by interaction. It simulates customer profiles, asset characteristics, market price movements (currently mock or from historical data), and customer responses to recommendations.
*   **State Representation:** The agent observes a state vector comprising:
    *   Customer's numerical risk level.
    *   Customer's numerical investment capacity.
    *   Percentage allocation of the customer's current portfolio to each available asset.
    *   `State = [risk_level, capacity, pct_asset_0, ..., pct_asset_N-1]`
*   **Action Space:** A discrete set of actions, where each action corresponds to recommending one of the `N` available financial assets.
*   **Dynamics:**
    *   At each step (representing one month), the agent recommends an asset.
    *   The simulated customer accepts this recommendation with a configurable probability.
    *   If accepted, a "Buy" transaction of a simulated value (based on customer capacity) is added to their portfolio.
    *   Time advances by one month.
*   **Reward Function:** The agent receives a reward based on the quality of its recommendation. The reward is a weighted sum of:
    1.  **Price Performance (`R_price`):** Calculated as the percentage return of the recommended asset over the past month.
        `R_price = (P_current - P_past) / P_past`
    2.  **Risk Match (`R_risk`):** Scores how well the recommended asset's volatility quartile (mapped to a 0-3 scale) matches the customer's risk profile (0-3).
        *   Perfect match: +1.0
        *   Close match (difference of 1): 0.0
        *   Mismatch (difference > 1): -1.0
    3.  **Diversification (`R_hhi`):** Measures the change in portfolio diversification using the Herfindahl-Hirschman Index (HHI). A decrease in HHI (better diversification) results in a positive reward component.
        `HHI = sum ( (value_of_asset_i / total_portfolio_value)^2 )`
        `R_hhi_score = - (HHI_after_recommendation - HHI_before_recommendation) * scaling_factor`
    *   The total reward is `(w_price*R_price + w_risk*R_risk + w_hhi*R_hhi) * global_reward_scale`. Weights and scale are configurable.

### 2. DQN Agent (`DQNAgent`)

*   **Algorithm:** Deep Q-Network (DQN), a value-based RL algorithm.
*   **Q-Network:** A neural network (Multi-Layer Perceptron) that takes the environment state as input and outputs an estimated Q-value (expected future cumulative reward) for each possible action (asset recommendation).
*   **Replay Buffer:** Stores past experiences `(state, action, reward, next_state, done)` to enable off-policy learning and break temporal correlations in training data.
*   **Learning Rule (Bellman Equation for Q-Learning):**
    The agent learns by minimizing the Mean Squared Error (MSE) between:
    *   The **target Q-value**: `Q_target = reward + gamma * max_a' Q_target_network(next_state, a')` (if not done)
    *   The **predicted Q-value**: `Q_policy_network(state, action)`
    A separate target network is used to stabilize learning.
*   **Action Selection:** Employs an epsilon-greedy strategy:
    *   With probability `1 - epsilon`: chooses the action with the highest Q-value (exploitation).
    *   With probability `epsilon`: chooses a random action (exploration). Epsilon decays over training.

### 3. Training Loop

*   The agent interacts with the environment for a configured number of episodes.
*   Within each episode, the agent observes states, takes actions, receives rewards, and stores experiences.
*   Periodically, the agent samples experiences from the replay buffer to update its Q-network.

### 4. Heuristic Baselines

To benchmark the RL agent's performance, several heuristic strategies are implemented (or planned):
*   **Random Policy:** Recommends an asset uniformly at random.
*   **Equal Weight (EW):** Aims to maintain an equal allocation across all assets.
*   **Mean-Variance Optimization (MVO):** Optimizes portfolio weights to maximize expected return for a given level of risk (volatility), based on historical price data. Requires `cvxpy`.
*   **Risk Parity (RP):** Allocates capital such that each asset contributes equally to the total portfolio risk, typically inversely proportional to its volatility.

### 5. Evaluation

*   A dedicated function (`evaluate_policy`) runs a specified policy (RL agent or heuristic) in the environment for a set number of episodes with no learning/exploration.
*   Performance metrics (average cumulative reward, standard deviation of rewards, average episode length) are collected.

### 6. Configuration (Hydra)

*   The project uses [Hydra](https://hydra.cc/) for managing configurations.
*   All hyperparameters and settings for the environment, agent, training, data loading, and evaluation are defined in YAML files within the `conf/` directory.
*   This allows for flexible experimentation and organized output management.

## Project Structure


financial_rl_project/
├── conf/ # Hydra configuration files
├── data/ # Raw data storage (e.g., CSVs)
├── notebooks/ # Jupyter notebooks for exploration
├── src/ # Source code
│ ├── agent/ # RL agent implementations
│ ├── data_management/ # Data loading and mock data generation
│ ├── environment/ # Financial environment simulation
│ ├── evaluation/ # Policy evaluation logic
│ ├── heuristics/ # Heuristic baseline policies
│ ├── training/ # Training loops and Optuna setup
│ ├── utils/ # Common utilities (plotting, seeding)
│ └── run.py # Main Hydra-enabled application script
├── tests/ # Unit and integration tests
├── outputs/ # Default Hydra output directory (auto-generated)
├── .gitignore
├── LICENSE
├── pyproject.toml # Project build and metadata configuration
├── README.md # This file
└── requirements.txt # Python dependencies

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd financial_rl_project
    ```
2.  **Create a Python virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional, for development) Install optional dependencies:**
    ```bash
    pip install -r requirements-dev.txt # If you create this from pyproject.toml
    # OR directly:
    # pip install pytest black flake8 mypy ipykernel
    ```
5.  **(Optional, for development) Install the project in editable mode:**
    This allows changes in `src/` to be immediately available without reinstalling.
    ```bash
    pip install -e .
    ```

## Running the Project

The main entry point is `src/run.py`, executed via the Python interpreter. Hydra manages the configuration.

**Basic Training Run (using default mock data):**
```bash
python src/run.py
```
