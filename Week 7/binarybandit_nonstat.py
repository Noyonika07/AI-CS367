import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. Non-Stationary 10-Armed Bandit Environment
# ---------------------------------------------
class NonStationaryBandit:
    def __init__(self, k=10):
        self.k = k
        self.means = np.zeros(k)  # all arms start equal
        self.walk_std = 0.01      # random walk std

    def step(self, action):
        # reward from chosen arm
        reward = np.random.normal(self.means[action], 1.0)

        # random walk for ALL means
        self.means += np.random.normal(0, self.walk_std, self.k)

        return reward


# -------------------------------------------------------
# 2. Modified Epsilon-Greedy Agent (constant step size)
# -------------------------------------------------------
class EpsilonGreedyAgent:
    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(k)  # estimated rewards

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.Q)

    def update(self, action, reward):
        self.Q[action] += self.alpha * (reward - self.Q[action])


# -------------------------
# 3. Run the experiment
# -------------------------
steps = 10000
bandit = NonStationaryBandit(k=10)
agent = EpsilonGreedyAgent(k=10, epsilon=0.1, alpha=0.1)

avg_rewards = []
total_reward = 0

for t in range(1, steps + 1):
    action = agent.select_action()
    reward = bandit.step(action)
    agent.update(action, reward)

    total_reward += reward
    avg_rewards.append(total_reward / t)

# -------------------------
# 4. Plot performance
# -------------------------
plt.figure(figsize=(12, 6))
plt.plot(avg_rewards, label="Average Reward", linewidth=1.5)
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Epsilon-Greedy Agent Performance in Non-Stationary Environment")
plt.legend()
plt.grid(True)
plt.show()
