import random
import math
import matplotlib.pyplot as plt

class PuzzleSimAnneal:
    def __init__(self, grid_size, edge_costs, start_temp=1200, min_temp=1e-4, cool_rate=0.992, max_steps=12000):
        """
        grid_size: dimension of the puzzle (grid_size x grid_size)
        edge_costs: dictionary mapping ((pieceA, pieceB), direction) -> mismatch value
        start_temp: initial temperature for annealing
        min_temp: lowest temperature threshold
        cool_rate: cooling multiplier per iteration
        max_steps: maximum number of iterations
        """
        self.grid_size = grid_size
        self.num_pieces = grid_size * grid_size
        self.edge_costs = edge_costs

        self.temperature = start_temp
        self.min_temp = min_temp
        self.cool_rate = cool_rate
        self.max_steps = max_steps

        # keep track of best result
        self.opt_state = None
        self.opt_cost = float('inf')
        self.progress = []

    def evaluate(self, layout):
        """Compute total border mismatch for a given layout."""
        cost = 0
        N = self.grid_size
        for row in range(N):
            for col in range(N):
                index = row * N + col
                piece = layout[index]

                # right neighbor
                if col < N - 1:
                    right_piece = layout[index + 1]
                    cost += self.edge_costs.get((piece, right_piece, 'right'), 0)

                # bottom neighbor
                if row < N - 1:
                    below_piece = layout[index + N]
                    cost += self.edge_costs.get((piece, below_piece, 'bottom'), 0)
        return cost

    def swap_random(self, layout):
        """Pick two random indices and swap them."""
        x, y = random.sample(range(len(layout)), 2)
        layout[x], layout[y] = layout[y], layout[x]
        return layout

    def solve(self):
        """Run simulated annealing to find near-optimal arrangement."""
        # start with a random arrangement
        current = list(range(self.num_pieces))
        random.shuffle(current)
        current_cost = self.evaluate(current)

        self.opt_state = current[:]
        self.opt_cost = current_cost
        self.progress.append(current_cost)

        iteration = 0
        while self.temperature > self.min_temp and iteration < self.max_steps:
            candidate = current[:]
            candidate = self.swap_random(candidate)
            candidate_cost = self.evaluate(candidate)

            delta = candidate_cost - current_cost
            # Accept better or probabilistically worse states
            if delta < 0 or random.random() < math.exp(-delta / self.temperature):
                current = candidate
                current_cost = candidate_cost
                if current_cost < self.opt_cost:
                    self.opt_state = current[:]
                    self.opt_cost = current_cost

            # gradually reduce temperature
            self.temperature *= self.cool_rate
            self.progress.append(self.opt_cost)
            iteration += 1

        print(f"✅ Best mismatch cost: {self.opt_cost}")
        print("🧩 Best puzzle configuration found:")
        for r in range(self.grid_size):
            print(self.opt_state[r*self.grid_size:(r+1)*self.grid_size])

    def plot_progress(self):
        """Visualize the cost decrease over time."""
        plt.plot(self.progress)
        plt.title("Simulated Annealing — Cost Evolution")
        plt.xlabel("Iteration")
        plt.ylabel("Total edge mismatch")
        plt.grid(True)
        plt.show()


# ----------------------
# Example Usage
# ----------------------

N = 3  # 3x3 puzzle
edge_costs = {}

# create dummy mismatch data for edges
for i in range(N*N):
    for j in range(N*N):
        edge_costs[(i, j, 'right')] = random.randint(0, 10)
        edge_costs[(i, j, 'bottom')] = random.randint(0, 10)

solver = PuzzleSimAnneal(N, edge_costs)
solver.solve()
solver.plot_progress()
