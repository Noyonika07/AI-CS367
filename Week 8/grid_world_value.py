import numpy as np

# Grid-World parameters
rows, cols = 3, 4
terminal_states = {(0, 3): 1.0, (1, 3): -1.0}
walls = {(1, 1)}
actions = ["U", "D", "L", "R"]
discount = 0.99

# Transition model probabilities
P_intended = 0.8
P_side = 0.1

# Step reward will be changed for different experiments
def value_iteration(step_reward, threshold=1e-4):
    V = np.zeros((rows, cols))

    # initialize terminal states
    for s, r in terminal_states.items():
        V[s] = r

    def in_grid(x, y):
        return 0 <= x < rows and 0 <= y < cols and (x, y) not in walls

    def move(x, y, a):
        if a == "U": nx, ny = x - 1, y
        elif a == "D": nx, ny = x + 1, y
        elif a == "L": nx, ny = x, y - 1
        elif a == "R": nx, ny = x, y + 1
        return (nx, ny) if in_grid(nx, ny) else (x, y)

    def get_neighbors(a):
        if a == "U": return ["L", "R"]
        if a == "D": return ["L", "R"]
        if a == "L": return ["U", "D"]
        if a == "R": return ["U", "D"]

    while True:
        delta = 0
        newV = V.copy()

        for x in range(rows):
            for y in range(cols):
                if (x, y) in terminal_states or (x, y) in walls:
                    continue

                values = []
                for a in actions:
                    nx, ny = move(x, y, a)
                    value = P_intended * (step_reward + discount * V[nx, ny])

                    for sa in get_neighbors(a):
                        sx, sy = move(x, y, sa)
                        value += P_side * (step_reward + discount * V[sx, sy])

                    values.append(value)

                newV[x, y] = max(values)
                delta = max(delta, abs(newV[x, y] - V[x, y]))

        V = newV
        if delta < threshold:
            break

    return V


# Running for all step rewards
for r in [-2, 0.1, 0.02, 1]:
    print(f"\n=== Value Function for r(s) = {r} ===")
    print(value_iteration(r))
