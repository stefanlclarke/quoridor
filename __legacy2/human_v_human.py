from pygame_.pygame_player import PygamePlayer

game = PygamePlayer(3, 1)
game.game.reset(random_positions=True)
game.play()
