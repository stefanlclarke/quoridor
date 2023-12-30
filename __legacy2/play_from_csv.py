import numpy as np
from game.game import Quoridor
from game.game.move_reformatter import move_reformatter

moves_p1 = np.genfromtxt('game_samples/play600_p1.csv', delimiter=",")
moves_p1 = moves_p1.reshape((moves_p1.size // 12, 12))
moves_p2 = np.genfromtxt('game_samples/play600_p2.csv', delimiter=",")
moves_p2 = moves_p2.reshape((moves_p2.size // 12, 12))

print(moves_p1.shape)
print(moves_p2.shape)


def play_from_csv(moves_p1, moves_p2):
    game = Quoridor()

    game.print()
    for i in range(len(moves_p1)):
        game.move(move_reformatter(moves_p1[i]))
        game.print()
        if len(moves_p2) >= i + 1:
            game.move(move_reformatter(moves_p2[i], flip=True))
            game.print()


play_from_csv(moves_p1, moves_p2)