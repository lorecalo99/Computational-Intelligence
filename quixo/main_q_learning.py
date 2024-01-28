import random
from game import Game, Move, Player
import numpy as np
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import pickle
import time 


class GameRL(Game):
    def __init__(self) -> None:
        super().__init__()
        valid_take = [(i, j) for i in range(0, 5) for j in range(0, 5) if i == 0 or i == 4 or j == 0 or j == 4]
        self.valid_moves = set()
        for i, j in valid_take:
            # left column -> can go RIGHT
            if i == 0:
                self.valid_moves.add(((i, j), Move.RIGHT))
                if j != 0:
                    self.valid_moves.add(((i, j), Move.TOP))
                if j != 4:
                    self.valid_moves.add(((i, j), Move.BOTTOM))
            # last column -> can go LEFT
            if i == 4:
                self.valid_moves.add(((i, j), Move.LEFT))
                if j != 0:
                    self.valid_moves.add(((i, j), Move.TOP))
                if j != 4:
                    self.valid_moves.add(((i, j), Move.BOTTOM))
            # first row -> can go BOTTOM
            if j == 0:
                self.valid_moves.add(((i, j), Move.BOTTOM))
                if i != 0:
                    self.valid_moves.add(((i, j), Move.LEFT))
                if i != 4:
                    self.valid_moves.add(((i, j), Move.RIGHT))
            # last row -> can go TOP
            if j == 4:
                self.valid_moves.add(((i, j), Move.TOP))
                if i != 0:
                    self.valid_moves.add(((i, j), Move.LEFT))
                if i != 4:
                    self.valid_moves.add(((i, j), Move.RIGHT))


    def play(self, player1: Player, player2: Player) -> int:
        '''play the game, returns the winning player and update the Q_table'''
        players = [player1, player2]
        winner = -1
        states = []
        actions = []
        boards = []
        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players) # player 0 starts
            
            ok = False
            # deepcopy because otherwise when I pass (state,action) to update the Q_value, the board is not coherent but already one move ahead
            board = deepcopy(self._board) 
            while not ok:
                from_pos, slide = players[self.current_player_idx].make_move(
                    self)
                ok = self._Game__move(from_pos, slide, self.current_player_idx)
            winner = self.check_winner()
            
            # to update the q_table I need the current state(n),action(n) and state(n+2)
            if self.current_player_idx == 0:
                boards.append(board)
                states.append(map_state(board, self.current_player_idx))
                actions.append(map_action(board, from_pos, slide))

            # keep only the data needed to update Q_table and intermediate rewards
            if len(states) > 2:
                states = states[1:]
                actions = actions[1:]
                boards = boards[1:]

            # avoid update after first move (I need at least the random response state)
            if len(states) == 2 and self.current_player_idx == 0:
                player1.update_q_table(boards, self.valid_moves, states, actions[0], winner)
                ''' ^^^ quick explanation for the Q_table update ^^^
                boards contains: [board before myP action, board before myP next action] -> needed to calculate intermediate rewards
                states contains: [state of myP, state of myP after random_player action]
                actions[0] contains: action done by myP '''
        return winner


    def play_without_training(self, player1: Player, player2: Player) -> int:
        ''' play the game and returns the winning player without updating the Q_table '''
        players = [player1, player2]
        winner = -1
        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            while not ok:
                from_pos, slide = players[self.current_player_idx].make_move(
                    self)
                ok = self._Game__move(from_pos, slide, self.current_player_idx)
            winner = self.check_winner()
        return winner
    

class RandomPlayer(Player):
    ''' player class to play randomly '''
    def __init__(self) -> None:
        super().__init__()


    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


class ManualPlayer(Player):
    ''' player class to play manually quixo '''
    def __init__(self) -> None:
        super().__init__()


    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        game.print()

        while(True):
            input_from_pos = input("Insert the position of your move: (row, column) ")
            row, column = map(int, input_from_pos.split(','))
            from_pos = (row, column)
            if from_pos in [(i, j) for i in range(0, 5) for j in range(0, 5) if i == 0 or i == 4 or j == 0 or j == 4]:
                break
            else:
                print("Invalid input, it must be of the type int,int and on the perimeter of the board")

        input_move = ''
        while True:
            input_move = input("Insert the slide direction: (0=top / 1=bottom / 2=left / 3=right) ")      
            if input_move in ['0', '1', '2', '3']:
                if input_move == '0':
                    move = Move.TOP
                if input_move == '1':
                    move = Move.BOTTOM                   
                if input_move == '2':
                    move = Move.LEFT
                if input_move == '3':
                    move = Move.RIGHT             
                break
            else:
                print("Invalid input, it must be of the type: 0/1/2/3")
        return from_pos, move
    

class Q_learning_Player(Player):
    def __init__(self, epsilon: float, Q: np.array=None) -> None:
        super().__init__()
        if not Q:
            self.Q = defaultdict(float)
        else:
            self.Q = Q
        self.epsilon = epsilon
    

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        current_player_idx = game.get_current_player()
        available_moves = [((i, j), slide) for (i, j), slide in game.valid_moves if game._board[j, i] == current_player_idx or game._board[j, i] < 0]
        #available_moves.sort(key = lambda x: (x[0], x[1].value))
        available_moves_map = [map_action(game._board, available_move[0], available_move[1]) for available_move in available_moves]

        if np.random.random() < self.epsilon:
            random_move = random.choice(available_moves)
            from_pos = random_move[0]
            move = random_move[1]
        else:
            # generate action a as the best action we can take in state s
            s = map_state(game._board, current_player_idx)
            # take the best action according to the q-value
            q_star_index = np.argmax([self.Q[(s, a)] for a in available_moves_map]) # insert also in the Q_table all the state-action met
            # extract pos and slide from the best action
            from_pos = available_moves[q_star_index][0]
            move = available_moves[q_star_index][1]
        return from_pos, move
    
    
    def update_q_table(self, boards, valid_moves, states, current_action, winner):
        ''' update the q_table (using model-free Q_learning) '''
        # Compute the best action possible on the next state
        current_player_idx = 0
        available_moves = [((i, j), slide) for (i, j), slide in valid_moves if boards[1][j, i] == current_player_idx or boards[1][j, i] < 0]
        available_moves_map = [map_action(boards[1], available_move[0], available_move[1]) for available_move in available_moves]
        q_star_index = np.argmax([Q[(states[1], a)] for a in available_moves_map])
        from_pos = available_moves[q_star_index][0]
        move = available_moves[q_star_index][1]
        best_next_a = map_action(boards[1], from_pos, move)
        
        if winner == 0:
            reward = 1
        elif winner == 1:
            reward = -1
        else:
            reward = self.compute_reward(boards)

        Q[(states[0], current_action)] = (1 - ALPHA) * Q[(states[0], current_action)] +\
            ALPHA * (reward + DISCOUNT_FACTOR * Q[(states[1], best_next_a)])
    

    def compute_reward(self, boards):
        ''' compute the reward that in Q_learning consists in the immediate reward after taken action a in state s '''
        tot_reward = 0
        
        # for the first move doesn't make sense to compute how far you are from the solution
        if max_same_value(boards[0],0) == 0:
            return tot_reward
        '''
        print("//////////////////////////////")
        print(f"\n{boards[0]}")
        print(max_same_value(boards[0],0))
        print(f"\n{boards[1]}")
        print(max_same_value(boards[1],0))
        '''
        '''
        # check if increase chanche of winning for the opponent -> |discarded|
        if max_same_value(boards[0],1) < max_same_value(boards[1],1):
            tot_reward -= 0.1
        '''
        # check if it increases chance of winning
        if max_same_value(boards[0],0) < max_same_value(boards[1],0):
            tot_reward += 0.01
        # check if it decreases chanche of winning for the opponent
        if max_same_value(boards[0],1) > max_same_value(boards[1],1):
            tot_reward += 0.01
        
        return tot_reward
        

def max_same_value(matrix, value):
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


def map_state(board:np.array, current_player_idx:int) -> tuple[frozenset, frozenset]:
    ''' map the board to two frozen sets with the positions of each player's symbols(0/1) '''
    other_player_idx = 1 - current_player_idx

    # get all the positions where the players have their pieces
    s1 = np.argwhere(board==current_player_idx)
    s2 = np.argwhere(board==other_player_idx)

    # map the positions to an integer in the range [0, 24]
    s1 = map(lambda x: x[1] * board.shape[1] + x[0], s1)
    s2 = map(lambda x: x[1] * board.shape[1] + x[0], s2)

    return (frozenset(s1), frozenset(s2))


def map_action(board: np.array, from_pos: tuple[int, int], slide: Move) -> tuple[int, int]:
    ''' map the position to an integer in the range [0, 24] '''
    from_pos = from_pos[1] * board.shape[1] + from_pos[0]
    return (from_pos, slide)


def are_symmetric(a: np.array, b: np.array):
    ''' check the symmetry (not used because too much computationally expensive) '''
    for i in range(4):
        for j in range(2):
            if np.array_equal(np.flip(a, j), b):
                return True
        a = np.rot90(a)
    return False



if __name__ == '__main__':

    EPSILON = 0.3
    ALPHA = 0.9
    DISCOUNT_FACTOR = 0.9
    EPISODES = 250_000 
    ''' minimum advised to see some results, around one hour and a half of training time;
    once the Q-table is achieved the results are consistent in all the tests, but with only
    250k episodes, obtaining a good Q-table depends on the run and sometimes fails '''

    LOAD_Q_TABLE = False
    SAVE_Q_TABLE = False  # note that saving a Q_Table obtained with more than 100k/200k episodes may lead to memory error 
    filename_Qtable = 'QTable.pkl'                                                 # depending on the used pc performances
    filename_Results = 'Results_250kEpisodes_Eps03_new.txt'

    # LOAD THE Q_TABLE
    if LOAD_Q_TABLE:
        print("Loading the Q_table..")
        with open(filename_Qtable, 'rb') as f:
            Q = pickle.load(f)
            player1 = Q_learning_Player(EPSILON, Q)
    else:
        player1 = Q_learning_Player(EPSILON)
        Q = player1.Q
    player2 = RandomPlayer()

    # TRAINING
    print("Training..")
    start_time = time.perf_counter()
    for i in tqdm(range(EPISODES)):
        g = GameRL()
        # play and update the Q_table
        winner = g.play(player1, player2)
        #if i == 500: 
        ''' to undersand the speed at which the player concludes the first matches and learns how to play;
            -> usefull to me to early interrupt runs, because the initial direction taken by the player, given 
            the fact that I can't physically do milions episodes, is strongly correlated to the final performances. 
            In my case (for each change I did) I always discarded runs in which the time to do the training 
            was greater than 5 seconds, but it's too much device-dependent to give general indications '''
            #print(f"\n{time.perf_counter()-start_time}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for the training = {elapsed_time}")
    print(f"Lenght Q_table = {len(Q)}")

    with open(filename_Results, 'a') as f_r:
        f_r.write(f"-- Parameters -- \n Epsilon = {EPSILON} \n Alpha = {ALPHA} \n Discount factor = {DISCOUNT_FACTOR} \n ")
        f_r.write(f"\n -- Details -- \n Number of episodes = {EPISODES} \n Resulting Q-Table lenght = {len(Q)} \n")
        f_r.write(f" Elapsed time for the training = {elapsed_time}\n")


    # TEST 
    print("\nTesting against a random player..")
    n_test = 5
    n_games_per_test = 1000

    with open(filename_Results, 'a') as f_r:
        f_r.write(f"\n-- Results for {n_games_per_test} games against a Random Player (repeated {n_test} times to show reproducibility) --\n")
        f_r.write("Trained agent plays first:\n")
    # test for trained agent playing first
    print("\nTrained agent plays first:")
    for _ in range(n_test):
        EPSILON = 0
        wins = 0
        losses = 0    

        player1 = Q_learning_Player(EPSILON, Q)
        player2 = RandomPlayer()

        with open(filename_Results, 'a') as f_r:
            for i in range(n_games_per_test):
                g = GameRL()
                winner = g.play_without_training(player1, player2)
                if winner == 0:
                    wins += 1
                elif winner == 1:
                    losses += 1
            f_r.write(f" Wins: {wins}, Losses: {losses} \n")
            print(f"Wins: {wins}, Losses: {losses}")

    # test for trained agent playing second
    with open(filename_Results, 'a') as f_r:
        f_r.write("Trained agent plays second:\n")
    print("\nTrained agent plays second:")       
    for _ in range(n_test):
        EPSILON = 0
        wins = 0
        losses = 0    

        player1 = RandomPlayer()
        player2 = Q_learning_Player(EPSILON, Q)

        with open(filename_Results, 'a') as f_r:
            for i in range(n_games_per_test):
                g = GameRL()
                winner = g.play_without_training(player1, player2)
                if winner == 1:
                    wins += 1
                elif winner == 0:
                    losses += 1
            f_r.write(f" Wins: {wins}, Losses: {losses} \n")
            print(f"Wins: {wins}, Losses: {losses}")


    # SAVE THE Q_TABLE (it works depending on the number of entries and on the memory of the used pc)
    if SAVE_Q_TABLE: 
        print("\nSaving the Q_table..")
        with open(filename_Qtable, 'wb') as f:
            pickle.dump(Q, f)
        print("Q_table saved")