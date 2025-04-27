# A custom autograder for this project

################################################################################
# A mini-framework for autograding
################################################################################

import optparse
import pickle
import random
import sys
import traceback
import json

class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass

class Tracker(object):
    def __init__(self, questions, maxes, prereqs, mute_output):
        self.questions = questions
        self.maxes = maxes
        self.prereqs = prereqs

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
*** because Question {} builds upon your answer for Question {}.
""".format(prereq, q, q, prereq))
                return False

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        import time
        print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
          print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (sum(self.points.values()),
            sum([self.maxes[q] for q in self.questions])))

        print("""
Your grades are NOT yet registered.  To register your grades, make sure
to follow your instructor's guidelines to receive credit on your project.
""")

    def add_points(self, pts):
        self.points[self.current_question] += pts

TESTS = []
PREREQS = {}
def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)

def test(q, points):
    def deco(fn):
        TESTS.append((q, points, fn))
        return fn
    return deco

def parse_options(argv):
    parser = optparse.OptionParser(description = 'Run public tests on student code')
    parser.set_defaults(
        edx_output=False,
        gs_output=False,
        no_graphics=False,
        mute_output=False,
        check_dependencies=False,
        )
    parser.add_option('--gradescope-output',
                        dest = 'gs_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_option('--mute',
                        dest = 'mute_output',
                        action = 'store_true',
                        help = 'Mute output from executing tests')
    parser.add_option('--question', '-q',
                        dest = 'grade_question',
                        default = None,
                        help = 'Grade only one question (e.g. `-q q1`)')
    (options, args) = parser.parse_args(argv)
    return options

def produceGradeScopeOutput(maxes, points, questions):
    out_dct = {}

    # total of entire submission
    total_possible = sum(maxes.values())
    total_score = sum(points.values())
    out_dct['score'] = total_score
    out_dct['max_score'] = total_possible
    out_dct['output'] = "Total score (%d / %d)" % (total_score, total_possible)
    out_dct['visibility'] = "visible"
    out_dct['stdout_visibility'] = "visible"
    # individual tests
    tests_out = []
    for name in questions:
      test_out = {}
      # test name
      test_out['name'] = name
      # test score
      test_out['score'] = points[name]
      test_out['max_score'] = maxes[name]
      # others
      is_correct = points[name] >= maxes[name]
      test_out['output'] = "  Question {num} ({points}/{max}) {correct}".format(
          num=(name[1] if len(name) == 2 else name),
          points=test_out['score'],
          max=test_out['max_score'],
          correct=('X' if not is_correct else ''),
      )
      test_out['tags'] = []
      tests_out.append(test_out)
    out_dct['tests'] = tests_out

    # file output
    with open('/autograder/results/results.json', 'w') as outfile:
    # with open('./results.json', 'w') as outfile:
        json.dump(out_dct, outfile)
    return

def main():
    options = parse_options(sys.argv)

    questions = set()
    maxes = {}
    for q, points, fn in TESTS:
        questions.add(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if options.grade_question:
        if options.grade_question not in questions:
            print("ERROR: question {} does not exist".format(options.grade_question))
            sys.exit(1)
        else:
            questions = [options.grade_question]
            PREREQS[options.grade_question] = set()

    tracker = Tracker(questions, maxes, PREREQS, options.mute_output)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            try:
                fn(tracker)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nCaught KeyboardInterrupt: aborting autograder")
                tracker.finalize()
                print("\n[autograder was interrupted before finishing]")
                sys.exit(1)
            except:
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()

    tracker.finalize()
    if(options.gs_output):
        produceGradeScopeOutput(tracker.maxes, tracker.points, tracker.questions)
################################################################################
# Tests begin here
################################################################################

from minesweeper import Minesweeper
from agent import QLearningAgent
from utils import read_bomb_map, Action, ActionType, Condition

from copy import deepcopy

bomb_map_1 = read_bomb_map("bomb_maps/map_1.txt")
rows_1, cols_1 = len(bomb_map_1), len(bomb_map_1[0])

bomb_map_2 = read_bomb_map("bomb_maps/map_2.txt")
rows_2, cols_2 = len(bomb_map_2), len(bomb_map_2[0])

bomb_map_3 = read_bomb_map("bomb_maps/map_3.txt")
rows_3, cols_3 = len(bomb_map_3), len(bomb_map_3[0])


@test('q1', points=2)
def question_1_reward(tracker):
    print("Evaluating Question 1: Reward function for Minesweeper")
    game = Minesweeper(rows=rows_1, cols=cols_1, bomb_map=bomb_map_1)
    fail_reward = game.reward(Condition.BOMB)
    win_reward = game.reward(Condition.WIN)
    in_progress_reward = game.reward(Condition.IN_PROGRESS)
    if fail_reward < 0:
        print("Passed Test Case 1, get 1 point")
        tracker.add_points(1)
    else:
        print("Failed Test Case 1")
    if win_reward > 0 and win_reward > in_progress_reward and in_progress_reward > fail_reward:
        print("Passed Test Case 2, get 1 point")
        tracker.add_points(1)
    else:
        print("Failed Test Case 2")

@test('q2', points=2)
def question_2_epsilon_greedy(tracker):
    print("Evaluating Question 2: Epsilon-greedy action selection")
    game = Minesweeper(rows=rows_1, cols=cols_1, bomb_map=bomb_map_1)
    agent = QLearningAgent(game)
    agent.epsilon = 1.0
    state = game.obs()
    action_set = set()
    for _ in range(100):
        action = agent.get_action(state)
        action_set.add((action.action_type, action.x, action.y))
    print(len(action_set))
    if len(action_set) > 1:
        print("Passed Test Case 1, get 1 point")
        tracker.add_points(1)
    else:
        print("Failed Test Case 1")
    agent.epsilon = 0.0
    state = game.obs()
    recorded_state = deepcopy(state)
    game.step(Action(ActionType.REVEAL, 0, 0))
    game.reset()
    agent.Q[agent.state_to_tuple(recorded_state)] = {Action(ActionType.REVEAL, 0, 0): 1}
    state = game.obs()
    action_set = set()
    for _ in range(100):
        action = agent.get_action(state)
        action_set.add((action.action_type, action.x, action.y))
    if len(action_set) == 1:
        print("Passed Test Case 2, get 1 point")
        tracker.add_points(1)
    else:
        print("Failed Test Case 2")

@test('q3', points=1)
def question_3_q_table_update(tracker):
    print("Evaluating Question 3: Q-table update")
    game = Minesweeper(rows=rows_1, cols=cols_1, bomb_map=bomb_map_1)
    agent = QLearningAgent(game)
    agent.alpha = 0.1
    agent.gamma = 0.9
    state = game.obs()
    recorded_state = deepcopy(state)
    action = Action(ActionType.REVEAL, 0, 0)
    reward = 10
    next_state, _, _ = game.step(action)
    agent.update_q_table(recorded_state, action, reward, next_state)
    state_tuple = agent.state_to_tuple(recorded_state)
    if state_tuple in agent.Q and action in agent.Q[state_tuple] and agent.Q[state_tuple][action] == 1:
        print("Passed Test Case 1, get 1 point")
        tracker.add_points(1)
    else:
        print("Failed Test Case 1")

@test('q4', points=1)
def question_4_simple_map(tracker):
    print("Evaluating Question 4: Simple bomb map")
    game = Minesweeper(rows=rows_1, cols=cols_1, bomb_map=bomb_map_1)
    agent = QLearningAgent(game)
    agent.train(episodes=1000)
    condition = agent.play()
    if condition == Condition.WIN:
        print("Passed Test Case 1, get 1 point")
        tracker.add_points(1)
    else:
        print("Failed Test Case 1")

@test('q5', points=4)
def question_5_bigger_map(tracker):
    print("Evaluating Question 5: Bigger bomb map")
    game = Minesweeper(rows=rows_2, cols=cols_2, bomb_map=bomb_map_2)
    agent = QLearningAgent(game, q_table_path="q_table_2.pkl")
    condition = agent.play()
    if condition == Condition.WIN:
        print("Passed Test Case 1, get 2 point")
        tracker.add_points(2)
    else:
        print("Failed Test Case 1")
    game = Minesweeper(rows=rows_3, cols=cols_3, bomb_map=bomb_map_3)
    agent = QLearningAgent(game, q_table_path="q_table_3.pkl")
    condition = agent.play()
    if condition == Condition.WIN:
        print("Passed Test Case 2, get 2 point")
        tracker.add_points(2)
    else:
        print("Failed Test Case 2")


if __name__ == '__main__':
    main()
