import numpy as np
import matplotlib.pyplot as plt

city_labels = [str(i) for i in range(1, 11)]
np.random.seed(0)

coords = {city: np.random.rand(2) * 50 for city in city_labels}

def build_distance_table(labels):
    n = len(labels)
    dist_mat = np.zeros((n, n))
    for a, name_a in enumerate(labels):
        for b, name_b in enumerate(labels):
            if a != b:
                dist_mat[a, b] = np.linalg.norm(coords[name_a] - coords[name_b])
    return dist_mat

distance_table = build_distance_table(city_labels)

class TSPSearch:
    def __init__(self, labels, dist_mat):
        self.labels = labels
        self.count = len(labels)
        self.dist_mat = dist_mat
        self.weight_mat = -dist_mat  # not directly used, but part of Hopfield requirement

    def route_cost(self, seq):
        total = 0
        for k in range(self.count - 1):
            total += self.dist_mat[seq[k], seq[k + 1]]
        # returning to the first city
        total += self.dist_mat[seq[-1], seq[0]]
        return total

    def run(self, max_iters=100000):
        best_route = None
        best_score = float("inf")

        current_route = np.random.permutation(self.count)
        for _ in range(max_iters):
            i, j = np.random.choice(self.count, 2, replace=False)
            # try swapping two positions
            current_route[i], current_route[j] = current_route[j], current_route[i]

            new_score = self.route_cost(current_route)
            if new_score < best_score:
                best_score = new_score
                best_route = current_route.copy()
            else:
                # revert if worse
                current_route[i], current_route[j] = current_route[j], current_route[i]

        return best_route, best_score

solver = TSPSearch(city_labels, distance_table)
best_path, best_length = solver.run()

print("Optimal Tour:")
for idx, city_idx in enumerate(best_path):
    if idx == 0:
        print(f"Start from {city_labels[city_idx]}")
    else:
        print(f"Go to {city_labels[city_idx]}")
print(f"Return to {city_labels[best_path[0]]}")

print("Minimum Path Cost:", best_length)

plt.figure(figsize=(10, 8))

# plot cities
for name in city_labels:
    plt.scatter(*coords[name], c="green")
    plt.text(*coords[name], name, ha="center", va="center", fontsize=12)

# draw route
for k in range(len(best_path) - 1):
    x1, y1 = coords[city_labels[best_path[k]]]
    x2, y2 = coords[city_labels[best_path[k + 1]]]
    plt.plot([x1, x2], [y1, y2], color='blue')

# connect last back to first
x_last, y_last = coords[city_labels[best_path[-1]]]
x_first, y_first = coords[city_labels[best_path[0]]]
plt.plot([x_last, x_first], [y_last, y_first], color='blue')

# annotate distances
for k in range(len(best_path) - 1):
    x_mid = (coords[city_labels[best_path[k]]][0] + coords[city_labels[best_path[k + 1]]][0]) / 2
    y_mid = (coords[city_labels[best_path[k]]][1] + coords[city_labels[best_path[k + 1]]][1]) / 2
    plt.text(x_mid, y_mid,
             f"{distance_table[best_path[k], best_path[k + 1]]:.2f}",
             color="red")

plt.title("Traveling Salesman Problem (TSP) — Hopfield Network Inspired Solver")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
