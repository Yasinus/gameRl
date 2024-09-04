import numpy as np
import random
import copy
import time
import numpy as np
import random
import copy
import time
import pygame
import torch


class Game2048:
    def __init__(self, size=4, pygame_enabled=True):
        self.size = size
        self.reset()

        # self.pygame_enabled = pygame_enabled
        # if self.pygame_enabled:
        #     pygame.init()
        #     self.screen = pygame.display.set_mode((400, 400))
        #     pygame.display.set_caption("2048 Game")

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.get_state()

    def get_state(self):
        board = self.board.copy()
        board[board == 0] = 1
        return np.log2(board).flatten()

    def add_random_tile(self):
        empty_tiles = [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]
        if empty_tiles:
            i, j = random.choice(empty_tiles)
            self.board[i, j] = 2 if random.random() < 0.9 else 4

    def move(self, direction):
        match direction:
            case 'up':
                self.board = np.rot90(self.board)
                self.board, score = self.move_left(self.board)
                self.board = np.rot90(self.board, -1)
                self.score += score
            case 'down':
                self.board = np.rot90(self.board, -1)
                self.board, score = self.move_left(self.board)
                self.board = np.rot90(self.board)
                self.score += score
            case 'left':
                self.board, score = self.move_left(self.board)
                self.score += score
            case 'right':
                self.board = np.fliplr(self.board)
                self.board, score = self.move_left(self.board)
                self.board = np.fliplr(self.board)
                self.score += score
            case _:
                raise ValueError('Invalid direction')
        self.add_random_tile()
        return self.get_state(), score

    def move_left(self, board):
        score = 0
        new_board = np.zeros_like(board)
        for i in range(self.size):
            j = 0
            k = 0
            while j < self.size:
                if board[i, j] == 0:
                    j += 1
                elif j == self.size - 1:
                    new_board[i, k] = board[i, j]
                    j += 1
                    k += 1
                else:
                    value = board[i, j]
                    l = j + 1
                    while l < self.size and board[i, l] == 0:
                        l += 1
                    if l < self.size and board[i, l] == value:
                        new_board[i, k] = 2 * value
                        score += 2 * value
                        j = l + 1
                    else:
                        new_board[i, k] = value
                        j = l
                    k += 1
        return new_board, score

    def is_game_over(self):
        for direction in ['up', 'down', 'left', 'right']:
            if self.can_move(direction):
                return False
        return True
    
    def can_move(self, direction):
        match direction:
            case 'up':
                board = np.rot90(self.board)
                return self.can_move_left(board)
            case 'down':
                board = np.rot90(self.board, -1)
                return self.can_move_left(board)
            case 'left':
                return self.can_move_left(self.board)
            case 'right':
                board = np.fliplr(self.board)
                return self.can_move_left(board)
            case _:
                raise ValueError('Invalid direction')
    
    def can_move_left(self, board):
        for i in range(self.size):
            had_zero = board[i, 0] == 0
            for j in range(1, self.size):
                if board[i, j] == 0:
                    had_zero = True
                    continue
                if board[i, j] == board[i, j - 1] or had_zero or (board[i, 0] == 0 and not had_zero):
                    return True
        return False



# Create an instance of the game and play it
 