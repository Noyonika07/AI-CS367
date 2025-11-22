def dfs(state, goal, path, visited):
    if state == goal:
        return path + [state]

    visited.add(state)
    index_empty = state.index('_')

    for move in possible_moves(state):
        if move not in visited:
            result = dfs(move, goal, path + [state], visited)
            if result:
                return result
    return None


def possible_moves(state):
    state = list(state)
    moves = []
    n = len(state)

    for i in range(n):
        if state[i] == 'E':  # move east (right)
            # move one step
            if i + 1 < n and state[i + 1] == '_':
                new_state = state.copy()
                new_state[i], new_state[i + 1] = new_state[i + 1], new_state[i]
                moves.append(''.join(new_state))
            # jump over one rabbit
            if i + 2 < n and state[i + 1] in ['W'] and state[i + 2] == '_':
                new_state = state.copy()
                new_state[i], new_state[i + 2] = new_state[i + 2], new_state[i]
                moves.append(''.join(new_state))

        elif state[i] == 'W':  # move west (left)
            # move one step
            if i - 1 >= 0 and state[i - 1] == '_':
                new_state = state.copy()
                new_state[i], new_state[i - 1] = new_state[i - 1], new_state[i]
                moves.append(''.join(new_state))
            # jump over one rabbit
            if i - 2 >= 0 and state[i - 1] in ['E'] and state[i - 2] == '_':
                new_state = state.copy()
                new_state[i], new_state[i - 2] = new_state[i - 2], new_state[i]
                moves.append(''.join(new_state))
    return moves


# Initial setup
start = "EEE_WWW"
goal = "WWW_EEE"
visited = set()
path = dfs(start, goal, [], visited)

if path:
    print("DFS Path found:")
    for step in path:
        print(step)
else:
    print("No solution found!")
