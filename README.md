An AI environment for the game "Quoridor" by Stefan Clarke.

Run "human_v_bot.py" to play a game (as red) against the AI.

RULES:
First player to get to the opposite side of the board from the one they begin on wins.
Game is played in turns.
On each turn a player can move one space (up, down, left, right) or place a 2x1 wall.
To move, click your player then click the square you wish to move to.
To place a wall, click the location you wish to place the wall.
You are not allowed to completely block yourself or your opponent off from their final
destination entirely.

The AI is a Q-Network trained by Sarsa(Lambda) which uses a tree-search during performance.
