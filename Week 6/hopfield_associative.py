import numpy as np
import matplotlib.pyplot as plt

def hebb_train(patterns):
    n_patterns, N = patterns.shape
    W = np.zeros((N, N))
    for p in range(n_patterns):
        W += np.outer(patterns[p], patterns[p])
    np.fill_diagonal(W, 0)
    W /= N
    return W

def recall(W, pattern, steps=2000):
    x = pattern.copy()
    N = len(x)
    for _ in range(steps):
        i = np.random.randint(N)
        h = W[i].dot(x)
        x[i] = 1 if h >= 0 else -1
    return x

N = 100  # 10×10 image
patterns = np.random.choice([1, -1], size=(5, N))
W = hebb_train(patterns)

# Pick pattern 0 and distort it
original = patterns[0].copy()
noisy = original.copy()

flip_idx = np.random.choice(N, size=15, replace=False)
noisy[flip_idx] *= -1     # add noise

# Recall
recalled = recall(W, noisy)

original_img = original.reshape(10, 10)
noisy_img = noisy.reshape(10, 10)
recalled_img = recalled.reshape(10, 10)

plt.figure(figsize=(12, 4))

# Original
plt.subplot(1, 3, 1)
plt.imshow(original_img, cmap='gray', interpolation='nearest')
plt.title("Original")
plt.axis('off')

# Noisy
plt.subplot(1, 3, 2)
plt.imshow(noisy_img, cmap='gray', interpolation='nearest')
plt.title("Noisy")
plt.axis('off')

# Recalled
plt.subplot(1, 3, 3)
plt.imshow(recalled_img, cmap='gray', interpolation='nearest')
plt.title("Recalled")
plt.axis('off')

plt.tight_layout()
plt.show()

matches = np.sum(original == recalled)
print(f"Recalled correctly: {matches}/100 bits")
