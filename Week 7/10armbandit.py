import numpy as np
import matplotlib.pyplot as plt

class NonStatBandit:
    def __init__(self, arms=10, eps=0.1, steps=1000):
        self.arms = arms
        self.eps = eps
        self.steps = steps
        
        # Estimated action-values
        self.q_values = np.zeros(arms)
        
        # How many times each arm was selected
        self.counts = np.zeros(arms)
        
        # True (non-stationary) means for rewards
        self.means = np.zeros(arms)
        
        # Track all rewards
        self.rewards = []

    def drift(self):
        """Random walk on all true reward means."""
        self.means += np.random.normal(0, 0.01, self.arms)

    def choose_action(self):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.eps:
            return np.random.randint(self.arms)
        return np.argmax(self.q_values)

    def step(self):
        """Run one interaction with the bandit."""
        action = self.choose_action()
        
        # Reward sampled from the arm's current mean
        reward = np.random.normal(self.means[action], 1)
        
        # Update counts & Q-values (sample-average update)
        self.counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.counts[action]
        
        # Non-stationary drift after each step
        self.drift()
        
        self.rewards.append(reward)
        return action, reward

    def run(self):
        """Simulate the whole sequence of steps."""
        for _ in range(self.steps):
            self.step()
        return self.q_values, self.rewards
    

bandit = NonStatBandit(arms=10, eps=0.1, steps=10000)
final_q, reward_list = bandit.run()

print("Final estimated Q-values:", final_q)
print("Total reward collected:", sum(reward_list))

# Plotting average rewards
cumulative = np.cumsum(reward_list)
avg_rewards = cumulative / (np.arange(1, len(reward_list) + 1))

plt.plot(avg_rewards, color='red', label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Steps')
plt.grid(True)
plt.legend()
plt.show()
