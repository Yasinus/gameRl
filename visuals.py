import pygame
from game2048 import Game2048
import torch


class Visual:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("2048 Game")


    def draw_board(self,game):
        self.screen.fill((255, 255, 255))
        for i in range(game.size):
            for j in range(game.size):
                value = game.board[i, j]
                color = self.get_tile_color(value)
                pygame.draw.rect(self.screen, color, (j * 100, i * 100, 100, 100))
                if value != 0:
                    font = pygame.font.Font(None, 36)
                    text = font.render(str(value), True, (0, 0, 0))
                    text_rect = text.get_rect(center=(j * 100 + 50, i * 100 + 50))
                    self.screen.blit(text, text_rect)
        pygame.display.flip()

    def get_tile_color(self, value):
        colors = {
            0: (68, 1, 84),
            2: (59, 82, 139),
            4: (33, 144, 141),
            8: (94, 201, 97),
            16: (253, 231, 37),
            32: (254, 187, 37),
            64: (254, 144, 35),
            128: (249, 100, 27),
            256: (238, 59, 20),
            512: (217, 33, 32),
            1024: (187, 16, 39),
            2048: (128, 0, 38)
        }
        return colors.get(value, (128, 0, 38))

    def play_game(self, game):
        while not game.is_game_over():
            self.draw_board(game)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    direction = None
                    match event.key:
                        case pygame.K_UP: 
                            direction = 'up'
                        case pygame.K_DOWN:
                            direction = 'down'
                        case pygame.K_LEFT:
                            direction = 'left'
                        case pygame.K_RIGHT:
                            direction = 'right'
                        case _:
                            pass

                    if direction is not None and game.can_move(direction):
                        game.move(direction)   

        print('Game over')
        print('Your score:', game.score)
        pygame.time.wait(2000)
        pygame.quit()


    def model_play(self, model,game):
        cannot_move_count = 0
        while not game.is_game_over():
            action = torch.argmax(model(torch.tensor(game.get_state(), dtype=torch.float32).unsqueeze(0))).item()
            game.move(['up', 'down', 'left', 'right'][action])
            self.draw_board(game)
            pygame.time.wait(500)
        print('Game over')
        print('Your score:', game.score)
        pygame.time.wait(2000)
        pygame.quit()

def main():
    visual = Visual()
    game = Game2048()
    visual.play_game(game)
    # visual.model_play(model,game)

if __name__ == '__main__':
    main()


