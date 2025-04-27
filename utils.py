from enum import Enum


class Cell(Enum):
    FLAGGED = -1
    REVEALED_BOMB = -2
    UNREVEALED = -3

class Condition(Enum):
    IN_PROGRESS = 'in_progress'
    WIN = 'win'
    BOMB = 'bomb'

class ActionType(Enum):
    REVEAL = 'reveal'
    FLAG = 'flag'

class Action:
    def __init__(self, action_type, x, y):
        self.action_type = action_type
        self.x = x
        self.y = y

def create_random_board(rows, cols, bombs):
        """
        Creates a random Minesweeper board.

        Args:
            rows (int): The number of rows in the board.
            cols (int): The number of columns in the board.
            bombs (int): The number of bombs to place on the board.

        Returns:
            list of list of str: A 2D list representing the game board, where ' ' indicates an empty cell and 'B' indicates a bomb.
        """
        import random
        board = [[' ' for _ in range(cols)] for _ in range(rows)]
        bomb_positions = random.sample(range(rows * cols), bombs)
        for pos in bomb_positions:
            x, y = divmod(pos, cols)
            board[x][y] = 'B'
        return board

def read_bomb_map(file_path):
    """
    Reads a bomb map from a file and returns it as a list of lists.

    Args:
        file_path (str): The path to the file containing the bomb map.

    Returns:
        list of list of str: A 2D list representing the bomb map, where each inner list is a row of the map.
    """
    with open(file_path, 'r') as file:
        bomb_map = [list(line.strip('\n')) for line in file.readlines()]
    return bomb_map

def read_clue_map(file_path):
    """
    Reads a clue map from a file and returns it as a list of lists.

    Args:
        file_path (str): The path to the file containing the clue map.

    Returns:
        list of list of int/str: A 2D list representing the clue map, where each inner list is a row of the map.
                                 Each cell is either an integer (number of neighboring bombs) or 'U' for unknown.
    """
    with open(file_path, 'r') as file:
        clue_map = []
        for line in file.readlines():
            row = []
            for char in line.strip('\n'):
                if char.isdigit():
                    row.append(int(char))
                else:
                    row.append(char)
            clue_map.append(row)
    return clue_map
