# This folder contains two solutions for the Quixo game.
Inside you can find:
- game.py that contains the basic functions to play Quixo
- main_MinMax.py that contains a solution using MinMax (my main proposed solution)
- main_q_learning.py that contains the code to training and saving a Q table using a Q_learning strategy 
- a pdf with the basic rules of Quixo

# Note
During the early stages of the project, I worked and talked about possible approaches to the problem with Luca Solaini, deciding an efficent way to represent states, actions and valid moves.

# MinMax (my main proposed solution)
In this file I have built a player that plays Quixo using a MinMax approach. Given a certain depth, he analyses the solution tree starting from the current state and then he chooses the best possible action. To choose the action I implemented the typical MinMax strategy, making my player choosing the best action for him maximizing a certain value and then making the opponent doing the same thing but minimizing the value. The algorithm computes the mentioned value as the difference between the max number of my player elements on a same row/column/main diagonal and the opponent ones. In the code this is managed recursively and thanks to the use of an explicit stack, so that I can keep track of the state at each level of the game tree. Furthermore, noticing the time required to play a game if the selected depth is higher, I have introduced a pruning option that can considerably shorten times.

# QLearning 
This file contains the code for training and testing a player using Q_learning techniques. In order to update the Q-table I used the Q-learning model free formula, where as Q_t (s',a') I considered the Q-value computed given the state after the action done by the random opponent (and not simply the state after the trained player's action), considering the best possible action from that state.
As rewards, I decided to set different values depending on how good the move of the trained player was, in particular I have chosen to analyse the following cases:
1) Final rewards:
	r = 1 if the trained player win
	r = -1 if the trained player loose
2) Intermediate rewards:
	r += 0.01 if the trained player increases the chances of winning
	r -= 0.01 if the trained player decreases the chances of winning of the opponent
To compute the intermediate rewards, I used the boards of two consecutive states, counting the maximum number of the given symbols (0/1) on the same row, column, or main diagonal. I have read on some paper that usually also a reward for doing a valid move could be given, but it was not useful in my case, since I have limited the possible moves only to the valid ones already from the beginning to make the training faster.
Regarding the parameters, I have done the following choices:
	α = 0.9 -> this the learning rate, which determines the extent to which new information overrides old information in the Q-table
	γ = 0.9 -> discount factor
	eps = 0.3 -> it defines the possibility of choosing a random action instead of looking at the Q-table, it allows the exploration of the solution space

# Final considerations
Although I found way more interesting working on the Q-learning method, I finally gave up to the idea that it was not surely the best approach to this problem. Indeed, to achieve some decent results during the tests, I had to do at least 250k episodes during the training, resulting in a Q-table of 70/80 milions entries. Therefore, this type of approach gave me different problems in terms of memory, and the real problem is that, also considering only valid moves, 80 milions entries were still not enough to see all the states and giving good results in any game. I leave the code in the repo because I think it could work better on large scale and on an high-performances hardware. It contains already the structure to load and save the Q-table in the case someone wants to split the training in different moments (also because it requires many hours).
On the other hand, MinMax was obviously quicker to write and test, and it gives also extremely good results also using a depth equal to 0. I expected that because I applied the same strategy used in Q-learning to evaluate if a state is good or not, but in this case the choice is in a certain sense deterministic, so it practically outplays the random player, giving most of the times a win rate of 100%.