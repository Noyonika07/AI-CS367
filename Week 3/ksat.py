import random

def generate_k_sat(k, m, n):
    """
    Generates a random k-SAT problem.
    Each clause contains k distinct variables or their negations.

    Parameters:
        k (int): Number of literals per clause
        m (int): Number of clauses
        n (int): Number of variables

    Returns:
        list: List of clauses (each clause is a list of integers)
    """
    if k > n:
        raise ValueError("k cannot be greater than n since variables must be distinct in a clause.")

    clauses = []
    for _ in range(m):
        # Randomly choose k distinct variables from n
        variables = random.sample(range(1, n + 1), k)

        # Randomly negate each variable with 50% chance
        clause = [var if random.choice([True, False]) else -var for var in variables]
        clauses.append(clause)

    return clauses


def print_k_sat(clauses):
    """
    Prints the generated k-SAT formula in a readable format.
    Example: (x1 ∨ ¬x3 ∨ x5) ∧ (¬x2 ∨ x4 ∨ ¬x6)
    """
    formula = []
    for clause in clauses:
        literals = [f"x{abs(v)}" if v > 0 else f"¬x{abs(v)}" for v in clause]
        formula.append("(" + " ∨ ".join(literals) + ")")
    print(" ∧ ".join(formula))


# --- Main execution ---
if __name__ == "__main__":
    # Input values
    k = int(input("Enter value of k (literals per clause): "))
    m = int(input("Enter number of clauses (m): "))
    n = int(input("Enter number of variables (n): "))

    # Generate and print random k-SAT instance
    clauses = generate_k_sat(k, m, n)
    print("\nGenerated Random k-SAT Formula:\n")
    print_k_sat(clauses)
