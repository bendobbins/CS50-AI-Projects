"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    taken = 0
    # Iterate through every value in board, determine turn based on even or odd number of non-empty spaces
    for row in board:
        for value in row:
            if value is not EMPTY:
                taken += 1
    if taken % 2 == 0:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    i = 0
    # Iterate through board, return a set of tuples of the coordinates of every empty space
    for row in board:
        j = 0
        for value in row:
            if value is EMPTY:
                actions.add((i, j))
            j += 1
        i += 1
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Create copy of board, check if move is valid, then make move and return new board if valid
    newBoard = copy.deepcopy(board)
    if newBoard[action[0]][action[1]] is not EMPTY:
        raise Exception("Move not valid")
    activePlayer = player(board)
    newBoard[action[0]][action[1]] = activePlayer
    return newBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Use helper functions to check for winner, return player that won if there is a winner
    win = three_in_a_row(winConditions(board))
    if win[0]:
        return win[1]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    counter = 0
    # Check if every board space is occupied or if there is a winner
    for row in board:
        for value in row:
            if value is not EMPTY:
                counter += 1
    if winner(board) or counter == 9:
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # Check for winner, return corresponding value if one player won, return 0 if not
    win = winner(board)
    if win == X:
        return 1
    if win == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # Check for terminal state (no moves to be made)
    if terminal(board):
        return None
    winningAction = []
    tyingAction = []
    activePlayer = player(board)

    def max_value(boardCopy):
        """
        Finds best action for ai by mapping winning actions to higher values and selecting best action based on highest value.
        """
        if terminal(boardCopy):
            # Make sure utility returns 1 for ai player, not human
            if activePlayer == X:
                return utility(boardCopy)
            return -utility(boardCopy)
        v = -math.inf
        # Check each possible action
        for action in actions(boardCopy):
            # Find best move for ai by finding move that leads to best utility (assuming human plays optimally)
            v = max(v, min_value(result(boardCopy, action)))
            # Only evaluate actions on main board
            if boardCopy == board:
                if v == 1:
                    # If an action wins, break loop and use that action
                    winningAction.append((action))
                if v == 0:
                    # If no winning action, any tie will work
                    if len(tyingAction) != 1:
                        tyingAction.append((action))
            # Alpha Beta pruning, don't check more actions if computer has an optimal action that will lead to a win
            if v == 1:
                break
        return v

    def min_value(boardCopy):
        """
        Finds best action for human by mapping winning actions to lower values and selecting best action based on lowest value.
        """
        if terminal(boardCopy):
            if activePlayer == X:
                # Again, make sure utility returns 1 for ai
                return utility(boardCopy)
            return -utility(boardCopy)
        v = math.inf
        for action in actions(boardCopy):
            # Find best move for human, then pass to ai
            v = min(v, max_value(result(boardCopy, action)))
            # Alpha Beta pruning, don't check any more actions for a board state if human has an optimal one that will lead to a win
            if v == -1:
                break
        return v

    if board != initial_state():
        # Find best action
        max_value(board)
        # If no tying or winning actions, choose randomly
        if not tyingAction and not winningAction:
            action = list(actions(board)).pop()
        # If winning action, return it
        elif winningAction:
            action = winningAction[0]
        # If no winning action, but tying action, return it
        else:
            action = tyingAction[0]
    # If board is in initial state, best first tic tac toe move for X is in a corner
    else:
        action = (0, 0)

    return action


def three_in_a_row(conditions):
    """
    Takes all possible win conditions for any tic tac toe board and returns whether a user has won or not, and if so, which user.
    """
    for condition in conditions:
        if condition[0] == condition[1] == condition[2] and condition[0] is not EMPTY:
            return (True, condition[0])
    return (False, "i am a useless variable")


def winConditions(board):
    """
    Returns values on current board for all possible ways that a player can win in tic tac toe.
    """
    return [
        [board[0][0], board[0][1], board[0][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][2], board[1][1], board[2][0]]
    ]

