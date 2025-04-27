import tkinter as tk
import platform

from utils import Action, ActionType, Cell, Condition

class MinesweeperUI:
    def __init__(self, game):
        self.game = game
        self.is_mac = platform.system() == "Darwin"
        self.buttons = [[None for _ in range(self.game.cols)] for _ in range(self.game.rows)]
        print(self.buttons)

        # Create the main game window
        self.root = tk.Tk()
        self.root.title("Minesweeper")
        self.message_label = tk.Label(self.root, text="", font=("Ariel", 14))
        self.message_label.grid(row=self.game.rows, column=0, columnspan=self.game.cols)
        self.create_buttons()

    def create_buttons(self):
        for x in range(self.game.rows):
            for y in range(self.game.cols):
                button = tk.Button(self.root, text='', width=4, height=2,
                                   command=lambda x=x, y=y: self.game.step(Action(ActionType.REVEAL, x, y)))
                button.grid(row=x, column=y)
                if self.is_mac:
                    button.bind('<Button-2>', lambda event, x=x, y=y: self.game.step(Action(ActionType.FLAG, x, y)))
                else:
                    button.bind('<Button-3>', lambda event, x=x, y=y: self.game.step(Action(ActionType.FLAG, x, y)))
                self.buttons[x][y] = button

    def update_gui(self, condition):
        # Update the cells
        for x in range(self.game.rows):
            for y in range(self.game.cols):
                button = self.buttons[x][y]
                # Update cell
                cell = self.game.revealed_board[x][y]
                if cell == Cell.REVEALED_BOMB:
                    button.config(text='💣', state='disabled', disabledforeground='red')
                elif cell == Cell.FLAGGED:
                    button.config(text='🚩', state='normal')
                elif isinstance(self.game.revealed_board[x][y], int):
                    if cell == 0:  # Revealed cell with no adjacent bombs
                        button.config(text='', state='disabled', disabledforeground='black')
                    elif cell > 0:  # Revealed cell with adjacent bombs
                        button.config(text=str(cell), state='disabled', disabledforeground='blue')
                else:  # Unrevealed cell
                    button.config(text='', state='normal')
                # Reset cell color
                if self.is_mac:
                    button.config(highlightbackground='SystemButtonFace')
                else:
                    button.config(bg='SystemButtonFace')
        # Highlight the last action
        if self.game.last_action:
            action = self.game.last_action
            button = self.buttons[action.x][action.y]
            if self.is_mac:
                button.config(highlightbackground='yellow')
            else:
                button.config(bg='yellow')
        # Update game condition based on the goal test
        if condition == Condition.BOMB:
            self.show_goal_test_msg("Game Over! You hit a bomb!")
        elif condition == Condition.WIN:
            self.show_goal_test_msg("Congratulations! You win!")
        elif condition == Condition.IN_PROGRESS:
            self.message_label.config(text="Keep going!")

    def show_goal_test_msg(self, message):
        self.message_label.config(text=message)
        for x in range(self.game.rows):
            for y in range(self.game.cols):
                self.buttons[x][y].config(state='disabled')

    def start_gui(self):
        self.root.mainloop()
