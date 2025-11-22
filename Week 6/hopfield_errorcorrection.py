import numpy as np
import matplotlib.pyplot as plt
import random

# Define patterns for letters D, J, C, M
patterns = {
    'D': np.array([-1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1]).reshape(5, 5),
    'J': np.array([-1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1]).reshape(5, 5),
    'C': np.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1]).reshape(5, 5),
    'M': np.array([-1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1]).reshape(5, 5)
}

# Display the patterns
plt.figure(figsize=(6, 6))
for idx, (letter, pattern) in enumerate(patterns.items()):
    plt.subplot(1, 4, idx + 1)
    plt.title(letter)
    plt.axis('off')
    plt.imshow(pattern, cmap='gray')
plt.show()

# Train the network using Hebbian learning
num_patterns = len(patterns)
pattern_size = np.prod(next(iter(patterns.values())).shape)
weights = np.zeros((pattern_size, pattern_size))

for pattern in patterns.values():
    flat_pattern = pattern.flatten()
    weights += np.outer(flat_pattern, flat_pattern)

np.fill_diagonal(weights, 0)
weights /= num_patterns

def introduce_errors(pattern, num_errors=1):
    """Introduce a specified number of errors into a pattern."""
    pattern_with_errors = pattern.copy()
    error_positions = set()
    while len(error_positions) < num_errors:
        i, j = np.random.randint(0, 5, size=2)
        if (i, j) not in error_positions:
            error_positions.add((i, j))
            pattern_with_errors[i, j] *= -1
    return pattern_with_errors

# Visualize error correction
plt.figure(figsize=(9, 45))
for error_count in range(1, 11):
    original_letter = random.choice(list(patterns.values()))
    noisy_letter = introduce_errors(original_letter, error_count)
    y = noisy_letter.flatten()

    # Iteratively correct errors
    previous_error = float('inf')
    current_error = np.linalg.norm(y)
    while current_error < previous_error:
        previous_error = current_error
        y = np.sign(weights @ y)
        current_error = np.linalg.norm(y - noisy_letter.flatten())

    # Plot results
    plt.subplot(10, 3, 3 * (error_count - 1) + 1)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(original_letter, cmap='gray')

    plt.subplot(10, 3, 3 * (error_count - 1) + 2)
    plt.title(f'With {error_count} error(s)')
    plt.axis('off')
    plt.imshow(noisy_letter, cmap='gray')

    plt.subplot(10, 3, 3 * (error_count - 1) + 3)
    plt.title('Corrected')
    plt.axis('off')
    plt.imshow(y.reshape(5, 5), cmap='gray')

plt.show()