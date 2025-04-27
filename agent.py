from utils import Condition

import pickle
from copy import deepcopy


class Agent:
    def __init__(self, game):
        """
        Initializes the agent with the given game instance.

        Args:
            game: An instance of the game that the agent will interact with.
        """
        self.game = game
        self.rows = game.rows
        self.cols = game.cols

    def play(self):
        """
        Executes the game loop for the agent.

        The agent continuously observes the game state, determines the next action,
        and performs the action until the game reaches a terminal condition.

        Returns:
            goal_test (Condition): The final state of the game, indicating whether
                                   the game is still in progress, won, or reveal a bomb.
        """
        raise NotImplementedError()

    def get_neighbors(self, x, y):
        """
        Get the neighboring coordinates of a given cell in a board.

        Args:
            x (int): The x-coordinate of the cell.
            y (int): The y-coordinate of the cell.

        Returns:
            list of tuple: A list of tuples representing the coordinates of the neighboring cells.
                           Only includes neighbors that are within the bounds of the board.
        """
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < self.game.rows and 0 <= y + dy < self.game.cols]


class ManualGuiAgent(Agent):
    def __init__(self, game):
        super().__init__(game)

    def play(self):
        pass


class QLearningAgent(Agent):
    def __init__(self, game, q_table_path="", alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.01):
        """
        Initializes the Q-learning agent with the given game instance.
         - alpha: Learning rate
         - gamma: Discount factor
         - epsilon: Exploration rate
         - epsilon_decay: Decay factor
         - min_epsilon: Minimum exploration
        """
        super().__init__(game)
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay factor
        self.min_epsilon = min_epsilon  # Minimum exploration
        # Initialize Q-table
        self.Q = {}  # Dictionary for state-action values
        if q_table_path:
            with open(q_table_path, "rb") as f:
                self.Q = pickle.load(f)

    def state_to_tuple(self, state):
        """
        Converts the given state to a tuple for hashing in the Q-table.
        """
        def convert(cell):
            return cell.value if hasattr(cell, 'value') else cell
    
        state_tuple = tuple(tuple(convert(cell) for cell in sublist) for sublist in state)
        return state_tuple

    def get_action(self, state):
        """
        Chooses an action using epsilon-greedy based on the given state.

        Args:
            state: The current state of the game.
        
        Returns:
            Action: The chosen action.
        """
        # TODO: Implement the epsilon-greedy action selection, using self.epsilon to switch strategy
        raise NotImplementedError()
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Updates the Q-table based on the given state, action, reward, and next state.

        Args:
            state (tuple): The current state of the game.
            action (Action): The action taken in the current state.
            reward (int): The reward received after taking the action.
            next_state (tuple): The resulting state after taking the action.
        """
        state_tuple = self.state_to_tuple(state)
        next_state_tuple = self.state_to_tuple(next_state)
        # TODO: Implement the Q-learning update rule, using the state_tuple as the key for the Q-table
        raise NotImplementedError()

    def train(self, episodes, save_path=""):
        """
        Trains the agent using Q-learning.

        Args:
            episodes (int): The number of episodes to train the agent.

        Returns:
            dict: The Q-table containing the state-action
                  values learned during training.
        """
        # Training loop
        print("Training Q-learning agent.")
        for _ in range(episodes):
            state = self.game.reset()
            condition = Condition.IN_PROGRESS
            while condition == Condition.IN_PROGRESS:
                old_state = deepcopy(state)
                action = self.get_action(state)
                next_state, condition, reward = self.game.step(action)
                self.update_q_table(old_state, action, reward, next_state)
                state = next_state
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        # Save the learned Q-table to file
        if save_path != "":
            with open(save_path, "wb") as f:
                pickle.dump(self.Q, f)
            print("Training complete! Q-table saved to", save_path)
        return self.Q

    def play(self):
        """
        Plays the game using the learned Q-table.

        Returns:
            Condition: The final state of the game, indicating whether
                       the game is still in progress, won, or reveal a bomb.
        """
        print("Playing Minesweeper using Q-learning agent.")
        state = self.game.reset()
        condition = Condition.IN_PROGRESS
        while condition == Condition.IN_PROGRESS:
            state_tuple = self.state_to_tuple(state)

            # TODO: Implement action selection for testing, choose the best action (do not need exploration)
            raise NotImplementedError()
        
            state, condition, _ = self.game.step(action)
        return condition
