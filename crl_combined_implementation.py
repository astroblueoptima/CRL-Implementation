
import random
import numpy as np
import matplotlib.pyplot as plt

# Grid and Environment setup
grid_size = 10
grid = np.zeros((grid_size, grid_size))
obstacle_positions = [(3, 3), (3, 4), (3, 5), (4, 5), (5, 5), (5, 4), (7, 7), (7, 6)]
for pos in obstacle_positions:
    grid[pos] = -1
start_pos = (0, 0)
end_pos = (grid_size-1, grid_size-1)
grid[start_pos] = 1
grid[end_pos] = 2

# Utility functions
def valid_actions(state, grid_size):
    actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    valid = []
    for action in actions:
        next_state = (state[0] + action[0], state[1] + action[1])
        if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
            valid.append(action)
    return valid

def step(state, action):
    next_state = (state[0] + action[0], state[1] + action[1])
    if next_state == end_pos:
        return next_state, 100
    if grid[next_state] == -1:
        return state, -100
    return next_state, -1

# Parameters
gamma = 0.9
alpha = 0.1
epsilon = 0.2
episodes = 500

# Initialize Q-values
Q_RL = {}
Q_CRL = {}
for i in range(grid_size):
    for j in range(grid_size):
        if (i, j) not in obstacle_positions:
            for action in valid_actions((i, j), grid_size):
                Q_RL[((i, j), action)] = 0
                Q_CRL[((i, j), action)] = 0

def choose_action(state, Q, mode="RL"):
    if random.uniform(0, 1) < epsilon:
        return random.choice(valid_actions(state, grid_size))
    else:
        values = {action: Q.get((state, action), 0) for action in valid_actions(state, grid_size)}
        if mode == "CRL":
            for action in values:
                next_state = (state[0] + action[0], state[1] + action[1])
                if next_state != end_pos and next_state not in obstacle_positions:
                    values[action] += gamma * max([Q.get((next_state, a), 0) for a in valid_actions(next_state, grid_size)])
        return max(values, key=values.get)

# Training both RL and CRL agents
rewards_RL = []
rewards_CRL = []

for episode in range(episodes):
    state = start_pos
    total_reward_RL = 0
    total_reward_CRL = 0
    
    while state != end_pos:
        # RL agent
        action_RL = choose_action(state, Q_RL)
        next_state_RL, reward_RL = step(state, action_RL)
        best_next_Q_RL = max([Q_RL.get((next_state_RL, a), 0) for a in valid_actions(next_state_RL, grid_size)])
        Q_RL[(state, action_RL)] = (1 - alpha) * Q_RL.get((state, action_RL), 0) + alpha * (reward_RL + gamma * best_next_Q_RL)
        state = next_state_RL
        total_reward_RL += reward_RL
        
        # CRL agent
        action_CRL = choose_action(state, Q_CRL, mode="CRL")
        next_state_CRL, reward_CRL = step(state, action_CRL)
        best_next_Q_CRL = max([Q_CRL.get((next_state_CRL, a), 0) for a in valid_actions(next_state_CRL, grid_size)])
        Q_CRL[(state, action_CRL)] = (1 - alpha) * Q_CRL.get((state, action_CRL), 0) + alpha * (reward_CRL + gamma * best_next_Q_CRL)
        state = next_state_CRL
        total_reward_CRL += reward_CRL

    rewards_RL.append(total_reward_RL)
    rewards_CRL.append(total_reward_CRL)

# Plotting the learning curves
plt.figure(figsize=(10, 5))
plt.plot(rewards_RL, label="RL", color="blue")
plt.plot(rewards_CRL, label="CRL", color="red")
plt.xlabel("Episodes")
plt.ylabel("Total Rewards")
plt.title("Learning Curves for RL vs. CRL")
plt.legend()
plt.show()
