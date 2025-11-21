# Gym-Environment-ML
A gym environment set to match Pokemon Gen 3's Sootopolis City Gym Ice Puzzle

This environment uses the Gym framework to create 3 seperate puzzles based on the Sootopolis City Gym ice puzzle. Each puzzle has a start, finish, as well as walls and rocks. The tiles that change are I, and once stepped on they become C for cracked. Just like in the actual gym, the puzzle can only be completed once all possible tiles are cracked, and the agent reaches the finishing tile. If the agent steps on a tile that is already cracked, the program ends. There is also spots for different statistics, such as % of successes, and the probability that the agent finishes all three puzzles (however this probability is ~0.00%). Currently, the tester is using a randomized agent that moves completely randomly. 

I am currently working to create a QLearning Agent to try and see if I can teach the agent to finish the puzzles quickly and accurately. 
