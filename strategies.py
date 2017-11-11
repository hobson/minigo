import copy
import math
import random
import sys
import time
import sgf_wrapper

import gtp
import numpy as np
from mcts import MCTSNode

import go
import utils

class MCTSPlayerMixin:
    def __init__(self, network, seconds_per_move=5, simulations_per_move=0,
                 resign_threshold=-0.999, verbosity=0, two_player_mode=False):
        self.network = network
        self.seconds_per_move = seconds_per_move
        self.simulations_per_move = simulations_per_move
        self.verbosity = verbosity
        self.two_player_mode = two_player_mode
        if two_player_mode:
            self.temp_threshold = -1
        else:
            self.temp_threshold = ((go.N * go.N) / 10) + 3
        self.searches_pi = []
        self.root = None
        self.result = 0
        self.resign_threshold = -abs(resign_threshold)
        super().__init__()

    def initialize_game(self):
        self.root = MCTSNode(go.Position())
        self.result = 0

    def suggest_move(self, position):
        ''' Used for playing a single game.
        For parallel play, use initialize_move, select_leaf,
        incorporate_results, and pick_move
        '''
        start = time.time()
        move_probs, value = self.network.run(position)
        self.root = MCTSNode(position)
        self.root.incorporate_results(move_probs, value)
        if not self.two_player_mode:
            self.root.inject_noise()

        if self.simulations_per_move == 0 :
            while time.time() - start < self.seconds_per_move:
                self.tree_search()
        else:
            while self.root.N < self.simulations_per_move:
                self.tree_search()

        if self.verbosity > 0:
            print("%d: Searched %d times in %s seconds\n\n" % (
                self.root.position.n, self.root.N, time.time() - start), file=sys.stderr)

        #print some stats on anything with probability > 1%
        if self.verbosity > 2:
            self.root.print_stats(sys.stderr)
            print('\n\n', file=sys.stderr)
            print(self.root.position, file=sys.stderr)

        return self.pick_move()

    def play_move(self, coords):
        '''
        Notable side effects:
          - finalizes the probability distribution according to
          this roots visit counts into the class' running tally, `searches`
          - Makes the node associated with this move the root, for future
            `inject_noise` calls.
        '''
        if not self.two_player_mode:
            self.searches_pi.append(
                self.root.children_as_pi(self.root.position.n > self.temp_threshold))
        self.root = self.root.add_child(utils.flatten_coords(coords))
        self.position = self.root.position # for showboard
        del self.root.parent.children
        return True # GTP requires positive result.

    def pick_move(self):
        '''Picks a move to play, based on MCTS readout statistics.

        Highest N is most robust indicator. In the early stage of the game, pick
        a move weighted by visit count; later on, pick the absolute max.'''
        if self.root.position.n > self.temp_threshold:
            fcoord = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            fcoord = cdf.searchsorted(selection)
            assert self.root.child_N[fcoord] != 0
        return utils.unflatten_coords(fcoord)

    def tree_search(self):
        leaf = self.root.select_leaf()
        move_probs, value = self.network.run(leaf.position)
        leaf.incorporate_results(move_probs, value, up_to=self.root)

    def is_done(self):
        '''True if the last two moves were Pass or if the position is at a move
        greater than (go.N^^2)*3.  False otherwise.
        '''
        if self.result != 0: #Someone's twiddled our result bit!
            return True

        if self.root.position.is_game_over():
            return True

        if self.root.position.n >= (go.N * go.N * 2):
            return True
        return False

    def show_path_to_root(self, node):
        pos = node.position
        if len(pos.recent) == 0:
            return
        moves = list(map(utils.to_human_coord,
                         [move.move for move in pos.recent[self.root.position.n:]]))
        print("From root: ", " <= ".join(moves), file=sys.stderr, flush=True)

    def should_resign(self):
        '''Returns true if the player resigned.  No further moves should be played'''
        if self.root.Q_perspective < self.resign_threshold: # Force resign
            self.result = self.root.position.to_play * -2 # use 2 & -2 as "+resign"
            if self.verbosity > 1:
                res = "B+" if self.result is 2 else "W+"
                print("%sResign: %.3f" % (res, self.root.Q))
                print(self.root.position, self.root.position.score())
            return True
        return False

    def make_result_string(self, pos):
        if abs(self.result) == 2:
            res = "B+Resign" if self.result == 2 else "W+Resign"
        else:
            res = pos.result()
        return res

    def to_sgf(self): 
        pos = self.root.position
        res = self.make_result_string(pos)
        return sgf_wrapper.make_sgf(pos.recent, res,
                                    white_name=self.network.name or "Unknown",
                                    black_name=self.network.name or "Unknown")

    def to_dataset(self):
        assert len(self.searches_pi) == self.root.position.n
        pwcs = list(go.replay_position(self.root.position))[:-1]
        results = np.ones([len(pwcs)], dtype=np.int8)
        if self.result < 0:
            results *= -1
        return (pwcs, self.searches_pi, results)
