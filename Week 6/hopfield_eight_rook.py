import numpy as np
import matplotlib.pyplot as plt

def create_random_board(n=8):
    """Create an n×n board with exactly n rooks placed randomly."""
    board = np.zeros((n, n), dtype=int)
    chosen_positions = np.random.choice(n * n, n, replace=False)
    for p in chosen_positions:
        row, col = divmod(p, n)
        board[row, col] = 1
    return board

def compute_energy(state):
    """Compute energy = row conflicts + column conflicts."""
    row_penalty = np.sum((state.sum(axis=1) - 1) ** 2)
    col_penalty = np.sum((state.sum(axis=0) - 1) ** 2)
    return row_penalty + col_penalty

def improve_board(state, steps=1000):
    """Hill-climbing attempt to reduce board conflicts."""
    energy_now = compute_energy(state)
    n = state.shape[0]

    for _ in range(steps):
        a, b = np.random.choice(n * n, 2, replace=False)
        r1, c1 = divmod(a, n)
        r2, c2 = divmod(b, n)

        if state[r1, c1] == 1 and state[r2, c2] == 0:
            # Try swapping placement
            state[r1, c1], state[r2, c2] = 0, 1
            updated_energy = compute_energy(state)

            # Accept move only if it lowers energy
            if updated_energy < energy_now:
                energy_now = updated_energy
            else:
                # Undo move
                state[r1, c1], state[r2, c2] = 1, 0

    return state, energy_now

initial = create_random_board()

plt.imshow(initial, cmap='binary', interpolation='nearest')
plt.title("Initial Board")
plt.show()

optimized, final_E = improve_board(initial)
print("Final optimized energy:", final_E)

plt.imshow(optimized, cmap='binary', interpolation='nearest')
plt.title("Optimized Board")
plt.show()
