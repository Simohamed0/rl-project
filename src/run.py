import numpy as np
import random
from collections import defaultdict

# Reproducibility
random.seed(42)
np.random.seed(42)

# Parameters
users = ['U1', 'U2', 'U3']
assets = ['Stock_A', 'Stock_B', 'Bond_C', 'ETF_D', 'Crypto_E']
num_assets = len(assets)
train_steps = 10
test_steps = 5
top_k = 2  # Hit Rate@K

# Simulate user preferences as latent weights
user_preferences = {
    u: np.random.dirichlet(np.ones(num_assets))  # Soft preference over assets
    for u in users
}

# Simulate transaction history (training + test)
history = defaultdict(list)

def sample_purchase(user):
    probs = user_preferences[user]
    return np.random.choice(assets, p=probs)

for user in users:
    for _ in range(train_steps + test_steps):
        purchase = sample_purchase(user)
        history[user].append(purchase)

# Define states: last purchased asset
states = assets + ['None']
actions = list(range(num_assets))  # Recommend one asset

# Initialize Q-table
Q = {(u, s): np.zeros(num_assets) for u in users for s in states}

# RL hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2

# --- Train on training history ---
for user in users:
    user_history = history[user][:train_steps]
    state = 'None'
    for purchase in user_history:
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[(user, state)])
        
        recommended_asset = assets[action]
        reward = 1 if recommended_asset == purchase else 0
        next_state = purchase

        Q[(user, state)][action] += alpha * (
            reward + gamma * np.max(Q[(user, next_state)]) - Q[(user, state)][action]
        )
        state = next_state

# --- Test with future history (Hit Rate@K) ---
hit_count = 0
total_tests = 0

for user in users:
    user_test_history = history[user][train_steps:]
    state = history[user][train_steps - 1] if train_steps > 0 else 'None'
    
    for true_purchase in user_test_history:
        q_values = Q[(user, state)]
        top_k_indices = np.argsort(q_values)[-top_k:][::-1]
        recommended_assets = [assets[i] for i in top_k_indices]

        if true_purchase in recommended_assets:
            hit_count += 1
        total_tests += 1
        state = true_purchase  # update state

hit_rate = hit_count / total_tests
print(f"Simulated Hit Rate@{top_k} = {hit_rate:.2f}")
