from graphics_display import MinesweeperUI
from utils import Action, ActionType, Cell, Condition, create_random_board


class Minesweeper:
    def __init__(self, rows=5, cols=5, bombs=5, bomb_map=None, gui=False):
        """
        Initializes the Minesweeper game.

        Args:
            size (int, optional): The size of the board (size x size). Defaults to 5.
            bombs (int, optional): The number of bombs to place on the board. Defaults to 5.
            bomb_map (list of list of int, optional): A predefined bomb map. If None, a random bomb map is created. Defaults to None.
            gui (bool, optional): Whether to initialize the game with a GUI. Defaults to False.
        """
        self.rows = rows
        self.cols = cols
        self.revealed_board = [[Cell.UNREVEALED for _ in range(cols)] for _ in range(rows)]

        # The actual bomp map, you should not access this variable and read it outside of this class
        self.__board = bomb_map if bomb_map else create_random_board(rows, cols, bombs)
        # For GUI purposes, you do not need to access and modify these variables.
        self.last_action = None  # Track the last revealed cell (x, y)
        self.gui = gui
        if self.gui:
            self.gui = MinesweeperUI(self)

    def obs(self):
        """
        Returns the current observation of the revealed board.

        The revealed board is a representation of the Minesweeper game board
        where cells that have been revealed are shown, and unrevealed cells
        remain hidden. Each cell can be one of the following values:
            - Cell.UNREVEALED: The cell has not been revealed.
            - Cell.FLAGGED: The cell has been flagged by the agent.
            - Cell.REVEALED_BOMB: The cell contains a bomb and has been revealed.
            - int: The cell has been revealed and contains a number indicating the

        Returns:
            list: A 2D list representing the revealed state of the Minesweeper board.
        """
        return self.revealed_board
    
    def reward(self, condition):
        """
        Returns the reward based on the game condition.

        Args:
            condition (Condition): The current condition of the game.

        Returns:
            int: The reward based on the given game condition.
        """
        # TOOD: Implement the reward function
        raise NotImplementedError()
    
    def actions(self, state):
        """
        Returns the list of valid actions that can be taken in the current state.

        The valid actions are the coordinates (x, y) of cells that have not been
        revealed or flagged.

        Returns:
            list: A list of valid actions that can be taken in the current state.
        """
        actions = []
        for x in range(self.rows):
            for y in range(self.cols):
                if state[x][y] == Cell.UNREVEALED:
                    actions.append(Action(ActionType.REVEAL, x, y))
        return actions

    def reset(self):
        """
        Resets the Minesweeper game to its initial state.

        Returns:
            list: The initial observation of the board after it is reset.
        """
        self.revealed_board = [[Cell.UNREVEALED for _ in range(self.cols)] for _ in range(self.rows)]
        self.last_action = None
        if self.gui:
            self.gui.update_gui(Condition.IN_PROGRESS)
        return self.obs()

    def step(self, action):
        """
        Takes a step in the Minesweeper game based on the given action.

        Args:
            action (Action): The action to be performed, which includes the type of action
                             (REVEAL or FLAG) and the coordinates (x, y) where the action
                             is to be performed.

        Returns:
            tuple: A tuple containing the current observation of the board, the game
                   condition after the action is performed, and the reward. The game
                   condition can be one of the following:
                   - Condition.IN_PROGRESS: The game is still ongoing.
                   - Condition.BOMB: The game is over because a bomb was revealed.
                   - Condition.WIN: The game is won because all non-bomb cells are revealed.

        Notes:
            - If the action is to reveal a cell and the cell contains a bomb, the game ends.
            - If the action is to flag a cell, the cell is marked as flagged or unflagged.
            - The method updates the GUI if it is enabled, otherwise it prints the board
              and the game status to the console.
        """
        x, y = action.x, action.y
        print(f"Action: {action.action_type} at ({x}, {y})")
        if isinstance(self.revealed_board[x][y], int) and self.revealed_board[x][y] >= 0:
            return self.obs(), Condition.IN_PROGRESS, self.reward(Condition.IN_PROGRESS)
        # Update the board based on the action
        if action.action_type == ActionType.REVEAL:
            if self.__board[x][y] == 'B':
                self.revealed_board[x][y] = Cell.REVEALED_BOMB
            else:
                self.reveal(x, y)
        elif action.action_type == ActionType.FLAG:
            self.revealed_board[x][y] = Cell.FLAGGED if self.revealed_board[x][y] != Cell.FLAGGED else Cell.UNREVEALED
        # Track the last action for highlighting
        self.last_action = action
        # Test if the game ends and update the GUI if necessary
        condition = self.goal_test()
        if self.gui:
            self.gui.update_gui(condition)
        else:
            self.print_board()
            if condition == Condition.BOMB:
                print("Game Over! You hit a bomb!")
            elif condition == Condition.WIN:
                print("Congratulations! You win!")
        return self.obs(), condition, self.reward(condition)
    
    def goal_test(self):
        """
        Tests the current state of the game to determine if the goal has been reached.

        Returns:
            Condition: The current condition of the game, which can be one of the following:
                - Condition.BOMB: If the last action revealed a bomb.
                - Condition.IN_PROGRESS: If there are still unrevealed or flagged cells that are not bombs.
                - Condition.WIN: If all non-bomb cells have been revealed.
        """
        action = self.last_action
        if action.action_type == ActionType.REVEAL and self.__board[action.x][action.y] == 'B':
            return Condition.BOMB
        for x in range(self.rows):
            for y in range(self.cols):
                if self.__board[x][y] != 'B' and self.revealed_board[x][y] in {Cell.UNREVEALED, Cell.FLAGGED}:
                    return Condition.IN_PROGRESS
        return Condition.WIN
    
    def print_board(self):
        """
        Prints the current state of the Minesweeper board.

        The board is printed row by row, with each cell represented by a specific character:
        - 'B' for revealed bombs.
        - 'F' for flagged cells.
        - A number (as a string) for revealed cells with adjacent bombs.
        - ' ' (space) for revealed cells with no adjacent bombs.
        - '.' for unrevealed cells.

        This method does not return any value; it only prints the board to the console.
        """
        for x in range(self.rows):
            row = []
            for y in range(self.cols):
                if self.revealed_board[x][y] == Cell.REVEALED_BOMB:
                    row.append('B')
                elif self.revealed_board[x][y] == Cell.FLAGGED:
                    row.append('F')
                elif isinstance(self.revealed_board[x][y], int):
                    if self.revealed_board[x][y] == 0:
                        row.append(' ')
                    elif self.revealed_board[x][y] > 0:
                        row.append(str(self.revealed_board[x][y]))
                else:
                    row.append('.')
            print(' '.join(row))
        print()

    def reveal(self, x, y, revealed=None):
        """
        Reveals the cell at the given coordinates (x, y) and recursively reveals adjacent cells if the cell has no adjacent bombs.

        Args:
            x (int): The x-coordinate of the cell to reveal.
            y (int): The y-coordinate of the cell to reveal.
            revealed (set, optional): A set of coordinates that have already been revealed. Defaults to None.

        Returns:
            None
        """
        if revealed is None:
            revealed = set()
        if (x, y) in revealed:
            return
        revealed.add((x, y))
        count = self.count_adjacent_bombs(x, y)
        self.revealed_board[x][y] = count
        if count == 0:
            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols:
                    self.reveal(nx, ny, revealed)

    def count_adjacent_bombs(self, x, y):
        """
        Counts the number of bombs adjacent to a given cell in the Minesweeper board.

        Args:
            x (int): The x-coordinate of the cell.
            y (int): The y-coordinate of the cell.

        Returns:
            int: The number of adjacent bombs.
        """
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        count = 0
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.__board[nx][ny] == 'B':
                count += 1
        return count
