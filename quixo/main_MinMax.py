################ MAIN SOLUTION ################

import random
from game import Game, Move, Player
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time


class MinmaxPlayer(Player):

    def __init__(self, player_idx: int, max_depth: int, pruning: bool):
        super().__init__()
        self.find_valid_moves()
        self.player_idx = player_idx
        self.max_depth = max_depth
        self.strategy = 1 # 1 -> max phase, 0 -> min phase
        self.pruning = pruning


    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        ''' makes the best possible move analyzing a MinMax tree of a given depth '''
        # I reduce the choises to only valid moves to speed up the algorithm
        available_moves = [((i, j), slide) for (i, j), slide in self.valid_moves if game._board[j, i] == self.player_idx or game._board[j, i] < 0]
        # evaluate all possible valid moves
        best_evaluation = -6 # looking for the max
        for action in available_moves:
            # use a copy of the game not to touch the actual board
            game_copy = deepcopy(game)
            game_copy._Game__move(action[0], action[1], self.player_idx)
            eval = self.minmax(1-self.player_idx, game_copy, self.max_depth, 1-self.strategy, best_evaluation)
            if eval > best_evaluation:
                best_evaluation = eval
                best_move = deepcopy(action)
        # return position and slide direction
        return best_move[0], best_move[1]
    

    def max_same_value(self, matrix:np.array, value:int) -> int:
        ''' counts the max number of the given symbols(0/1) on the same row, column or main diagonal '''
        cols = matrix.shape[1]
        # rows check
        max_row = max([sum(arr == value) for arr in matrix])
        # columns check
        max_col = max([np.sum(matrix[:, j] == value) for j in range(cols)])
        # diagonal 1 check
        main_diagonal = np.diagonal(matrix)
        max_main_diagonal = np.sum(main_diagonal == value)
        # diagonal 2 check
        anti_diagonal = np.diagonal(np.fliplr(matrix))
        max_anti_diagonal = np.sum(anti_diagonal == value)
        return max(max_row, max_col, max_main_diagonal, max_anti_diagonal)
    

    def state_evaluation(self, game: Game) -> int:
        ''' gives an idea of how good a state is for the MinMax player '''
        return(self.max_same_value(game._board, self.player_idx) - self.max_same_value(game._board, 1-self.player_idx))
    

    def minmax(self, current_player_idx: int, game: Game, depth: int, strategy: int, last_evaluation: int) -> int:
        ''' implementation of the MinMax '''

        # keeps track of the state at each level of the game tree
        stack = [(current_player_idx, game, depth, strategy, last_evaluation)]
        # continues as long as the stack is not empty
        while stack:
            current_player_idx, game, depth, strategy, last_evaluation = stack.pop()

            # if the current state is a terminal state
            if depth == 0 or game.check_winner() != -1:
                return self.state_evaluation(game)
            
            available_moves = [((i, j), slide) for (i, j), slide in self.valid_moves if game._board[j, i] == current_player_idx or game._board[j, i] < 0]
            if strategy == 1: # looking for the max
                best_evaluation = -6
            else: # looking for the min
                best_evaluation = 6
            
            # evaluate all possible valid moves
            for action in available_moves:
                game_copy = deepcopy(game)
                game_copy._Game__move(action[0], action[1], current_player_idx)
                eval = self.minmax(1-current_player_idx, game_copy, depth-1, 1-strategy, best_evaluation)
                
                if strategy == 1: # looking for the max
                    if eval > best_evaluation:
                        best_evaluation = eval
                else: # looking for the min
                    if eval < best_evaluation:
                        best_evaluation = eval
                
                if self.pruning: # pruning not to explore the whole tree
                    if (strategy == 1 and best_evaluation >= last_evaluation): 
                        return best_evaluation
                    if (strategy == 0 and best_evaluation <= last_evaluation):
                        return best_evaluation
            
                stack.append((1-current_player_idx, game_copy, depth-1, 1-strategy, best_evaluation))

        return best_evaluation   


class RandomPlayer(Player):
    ''' player class to play randomly '''
    def __init__(self) -> None:
        super().__init__()


    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move



if __name__ == '__main__':

    # choose the depth [0, 1, 2, 3, ...] and the number of games to test the MinMax player
    depth = 2
    n_games = 100
    pruning = True # True recommended choice, it saves a lot of time

    start_time = time.perf_counter()

    # TEST PLAYING FIRST
    print(f"\nTesting against a random player (playing first with depth={depth} and pruning={pruning})")
    wins = 0
    losses = 0

    player1 = MinmaxPlayer(player_idx = 0, max_depth = depth, pruning = pruning)
    player2 = RandomPlayer()

    for _ in tqdm(range(n_games)):
        g = Game()
        winner = g.play(player1, player2)
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
    print(f"Wins: {wins}, Losses: {losses}")


    # TEST PLAYING SECOND
    print(f"\nTesting against a random player (playing second with depth={depth} and pruning={pruning})")
    wins = 0
    losses = 0

    player1 = RandomPlayer()
    player2 = MinmaxPlayer(player_idx = 1, max_depth = depth, pruning = pruning)
    
    for _ in tqdm(range(n_games)):
        g = Game()
        winner = g.play(player1, player2)
        if winner == 1:
            wins += 1
        elif winner == 0:
            losses += 1
    print(f"Wins: {wins}, Losses: {losses}")


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nElapsed time = {elapsed_time}")