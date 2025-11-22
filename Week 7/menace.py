import random

class MENACE:
    def __init__(self):
        self.gameStateBoxes = {}
        self.random = random.Random()
        self.wins = 0
        self.losses = 0
        self.draws = 0

    # Get next move based on current state
    def getNextMove(self, currentState):
        if currentState not in self.gameStateBoxes:
            self.initializeState(currentState)

        possibleMoves = self.gameStateBoxes[currentState]

        if not possibleMoves:
            raise Exception("No available moves for state: " + currentState)

        # Create weighted list
        weightedMoves = []
        for move, weight in possibleMoves.items():
            weightedMoves.extend([move] * weight)

        return random.choice(weightedMoves)

    # Initialize state with weight 3 for empty cells
    def initializeState(self, state):
        possibleMoves = {}
        for i in range(9):
            if state[i] == '.':
                possibleMoves[i] = 3
        self.gameStateBoxes[state] = possibleMoves

    # Reinforce based on result
    def reinforce(self, state, move, result):
        moves = self.gameStateBoxes.get(state)
        if moves and move in moves:
            if result == "win":
                moves[move] += 1
            elif result == "lose":
                moves[move] = max(moves[move] - 1, 1)
            elif result == "draw":
                moves[move] += 1

    # Play one full game
    def playGame(self):
        currentState = "........."
        states = []
        moves = []
        menaceTurn = True

        print("<-----------Starting new game!------------>")
        self.printBoard(currentState)

        while not self.isGameOver(currentState):
            if menaceTurn:
                move = self.getNextMove(currentState)
                states.append(currentState)
                moves.append(move)
                currentState = self.applyMove(currentState, move, 'X')
                print("MENACE makes move:")
            else:
                available = self.getAvailableMoves(currentState)
                move = random.choice(available)
                currentState = self.applyMove(currentState, move, 'O')
                print("Random opponent makes move:")

            self.printBoard(currentState)
            menaceTurn = not menaceTurn

        # Get result and reinforce
        result = self.getResult(currentState)
        for st, mv in zip(states, moves):
            self.reinforce(st, mv, result)

        print("Game result: " + result)

        # Update counters
        if result == "win":
            self.wins += 1
        elif result == "lose":
            self.losses += 1
        elif result == "draw":
            self.draws += 1

    # Check if game is over
    def isGameOver(self, state):
        return self.getResult(state) is not None or '.' not in state

    # Apply move
    def applyMove(self, state, move, player):
        chars = list(state)
        chars[move] = player
        return "".join(chars)

    # Available moves
    def getAvailableMoves(self, state):
        return [i for i, c in enumerate(state) if c == '.']

    # Determine game result
    def getResult(self, state):
        winningPatterns = [
            "012", "345", "678",
            "036", "147", "258",
            "048", "246"
        ]

        for pattern in winningPatterns:
            a, b, c = map(int, pattern)
            first = state[a]
            if first != '.' and first == state[b] and first == state[c]:
                return "win" if first == 'X' else "lose"

        if '.' not in state:
            return "draw"

        return None

    # Print board
    def printBoard(self, state):
        for i in range(0, 9, 3):
            print(state[i] + " | " + state[i+1] + " | " + state[i+2])
        print()

    # Print results
    def printResults(self):
        print("Final Results:")
        print("MENACE Wins:", self.wins)
        print("MENACE Losses:", self.losses)
        print("Draws:", self.draws)


# MAIN
if __name__ == "__main__":
    menace = MENACE()
    for _ in range(5):
        menace.playGame()
    menace.printResults()
