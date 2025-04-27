import os
import threading
from optparse import OptionParser

from agent import *
from minesweeper import Minesweeper
from utils import read_bomb_map

def main():
    """
    Main function to parse command-line options and start the Minesweeper game with the specified agent.

    Command-line options:
    -a, --agent: Type of agent to use (manual or rule_based)
    -m, --map: Path to the bomb map file
    """
    parser = OptionParser()
    parser.add_option("-a", "--agent", dest="agent_type", help="Type of agent to use (manual|bayes)")
    parser.add_option("-m", "--map", dest="bomb_map_file", help="Path to the bomb map file")
    parser.add_option("-o", "--out_model", dest="output_model_path", default="", help="Output path for the learned Q-table")
    parser.add_option("-i", "--in_model", dest="input_model_path", default="", help="Input path for the Q-table pickle file")
    parser.add_option("-e", "--episodes", dest="episodes", type="int", default=1000, help="Number of episodes to train the agent")

    (options, args) = parser.parse_args()

    if not options.agent_type or not options.bomb_map_file:
        parser.print_help()
        return

    agent_type = options.agent_type.lower()
    bomb_map_file = options.bomb_map_file

    if not os.path.exists(bomb_map_file):
        print(f"Error: The bomb map file '{bomb_map_file}' does not exist.")
        return

    bomb_map = read_bomb_map(bomb_map_file)

    rows, cols = len(bomb_map), len(bomb_map[0])
    if agent_type == "manual":
        game = Minesweeper(rows=rows, cols=cols, bomb_map=bomb_map, gui=True)
        agent = ManualGuiAgent(game)
    elif agent_type == "rl":
        game = Minesweeper(rows=rows, cols=cols, bomb_map=bomb_map)
        agent = QLearningAgent(game, q_table_path=options.input_model_path)
        agent.train(episodes=options.episodes, save_path=options.output_model_path)
        agent.play()
    else:
        print("Unknown agent type. Use 'manual' or 'rule_based'.")
        return

    agent_thread = threading.Thread(target=agent.play)
    agent_thread.start()
    if game.gui:
        game.gui.start_gui()

if __name__ == "__main__":
    main()
