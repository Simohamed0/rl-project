# Financial Asset Recommendation Simulation


## 🧪 Experiment Overview

### ✅ **Objectives**

- Simulate a multi-day, multi-user recommendation environment.
- Evaluate **portfolio recommendation strategies**:
  - **Mean-Variance Optimization (MVO)**: Optimizes expected return under risk constraints.
  - **Risk Parity (RP)**: Allocates equal risk to all assets.
  - **Equal-Weight (EW)**: Allocates uniform weights to all assets.

- Analyze:
  - **Alignment** between recommendations and user portfolios (HR@k).
  - **Impact** of recommendations on user returns (P(R)@k, ΔR@k).
  - **Performance differences** between strategies.

---

## 🧠 Techniques Used

### 🏦 **Portfolio Construction**

- **MVO (Markowitz Portfolio Theory)**:
  - Solves a convex optimization problem for each user-day using `cvxpy`.
  - Objective: Maximize expected return subject to a variance constraint (user risk budget).

- **Risk Parity**:
  - Allocates weights inversely proportional to asset volatility.
  - Scaled to meet user risk constraints.

- **Equal-Weight**:
  - Uniform weights across all assets.

### 🔍 **Evaluation Framework**

For each user and each day:
- **Recommendations**: Rank assets by heuristic weights.
- **Portfolio Hit-Check**:
  - Instead of simulating purchases, we **check if the user already holds any of the top-k recommended assets** in their portfolio.
- **Metrics Computed**:
  - **HR@k**: Hit Ratio—was a recommended asset in the user’s portfolio?
  - **P(R)@k**: % users whose returns improved after recommendations.
  - **ΔR@k**: Average change in return from original portfolio to recommended portfolio.

---

## 📊 Data Generation

### 🏙️ **Market Data (Synthetic)**

- **10 financial assets**.
- Daily returns generated from a **multivariate normal distribution**:
  - Randomly sampled volatilities (0.6% - 2% per day).
  - Expected daily return ~0.05/252 (annualized ~5%).
  - Random covariance structure.

- 10 days of simulated price and return data.

### 👥 **User Data (Synthetic)**

- **120 users**:
  - Random **risk budgets** (volatility caps).
  - Random **investment capacities** ($10,000 - $150,000).
  - Random initial portfolios (60% sparsity) generated from Dirichlet distribution.

---

## 📊 Example Results

| Strategy      | Avg ROI (%) | Avg ROI ($)   |
|---------------|-------------|---------------|
| Equal Weight  | 0.0048      | 398.74        |
| MVO           | 0.0341      | 2824.67       |
| Risk Parity   | 0.0048      | 396.46        |

| Strategy      | k   | HR      | P(R)    | ΔR     |
|---------------|-----|----------|---------|--------|
| Equal Weight  | 3   | 0.7500   | 0.566   | 0.0017 |
| MVO           | 3   | 0.7667   | 0.9992  | 0.0311 |
| Risk Parity   | 3   | 0.7750   | 0.5683  | 0.0017 |

---

