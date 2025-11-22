import numpy as np
from functools import lru_cache
from math import exp, factorial

# Poisson with truncation
def poisson(lmbda, n):
    return (lmbda**n) * exp(-lmbda) / factorial(n)

def truncated_poisson(lmbda, limit=10):
    P = [poisson(lmbda, i) for i in range(limit)]
    P[-1] += 1 - sum(P)  # absorb tail probability
    return P

# Environment parameters
MAX_BIKES = 20
MOVE_LIMIT = 5
RENT_REWARD = 10
MOVE_COST = 2
PARK_PENALTY = 4
DISCOUNT = 0.9

RENT1, RENT2 = 3, 4
RET1, RET2 = 3, 2

# Precompute truncated Poisson distributions
P_RENT1 = truncated_poisson(RENT1)
P_RENT2 = truncated_poisson(RENT2)
P_RET1 = truncated_poisson(RET1)
P_RET2 = truncated_poisson(RET2)

V = np.zeros((MAX_BIKES+1, MAX_BIKES+1))
policy = np.zeros_like(V, dtype=int)

@lru_cache(None)
def expected_return(state, action):
    i, j = state

    # apply free transfer
    effective = action - 1 if action > 0 else action
    effective = np.clip(effective, -MOVE_LIMIT, MOVE_LIMIT)

    # updated inventory
    new1 = int(np.clip(i - effective, 0, MAX_BIKES))
    new2 = int(np.clip(j + effective, 0, MAX_BIKES))

    move_cost = abs(action - 1) * MOVE_COST if action > 0 else abs(action) * MOVE_COST

    parking_cost = (PARK_PENALTY if new1 > 10 else 0) + (PARK_PENALTY if new2 > 10 else 0)

    expected = -move_cost - parking_cost

    # Compute expected reward with truncated loops
    for r1, p1 in enumerate(P_RENT1):
        for r2, p2 in enumerate(P_RENT2):

            actual_rent1 = min(new1, r1)
            actual_rent2 = min(new2, r2)

            reward = RENT_REWARD * (actual_rent1 + actual_rent2)
            prob_rent = p1 * p2

            for t1, q1 in enumerate(P_RET1):
                for t2, q2 in enumerate(P_RET2):

                    next1 = min(MAX_BIKES, new1 - actual_rent1 + t1)
                    next2 = min(MAX_BIKES, new2 - actual_rent2 + t2)

                    expected += prob_rent * q1 * q2 * (reward + DISCOUNT * V[next1, next2])

    return expected


def policy_iteration():
    global V, policy

    stable = False
    while not stable:

        # Policy Evaluation (only 5 sweeps needed)
        for _ in range(5):
            newV = V.copy()
            for i in range(MAX_BIKES+1):
                for j in range(MAX_BIKES+1):
                    newV[i, j] = expected_return((i, j), policy[i, j])
            V[:] = newV

        # Policy Improvement
        stable = True
        for i in range(MAX_BIKES+1):
            for j in range(MAX_BIKES+1):
                old = policy[i, j]

                best_val = -1e9
                best_act = 0
                for a in range(-MOVE_LIMIT, MOVE_LIMIT+1):
                    val = expected_return((i, j), a)
                    if val > best_val:
                        best_val = val
                        best_act = a

                policy[i, j] = best_act
                if best_act != old:
                    stable = False

    return policy, V


# Run
policy, V = policy_iteration()

print("Optimal Policy:")
print(policy)
