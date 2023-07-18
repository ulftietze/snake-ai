import random
from collections import deque

import numpy
import torch

from model import LinearQNet, QTrainer
from plotting import plot
from snake_game import SnakeGameAI, Direction, Point

MAX_MEMORY = 10_000_000
BATCH_SIZE = 1_000
HIDDEN_NODES = 5_000
LEARNING_RATE = 0.005


class Agent:

    def __init__(self):
        self.numer_of_games = 0
        self.epsilon = 70  # randomness
        self.gamma = 0.9  # discount rate (needs to be smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY)  # call popleft() on double linked list if full
        # self.model = LinearQNet(782, 5000, 3) # Use if we include all fields (doesn't work very well -.-)
        self.model = LinearQNet(11, HIDDEN_NODES, 3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, is_game_over):
        self.memory.append((state, action, reward, next_state, is_game_over))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            minimal_sample = random.sample(self.memory, BATCH_SIZE)  # list of random tuples when memory is empty
        else:
            minimal_sample = self.memory

        states, actions, rewards, next_states, is_game_overs = zip(*minimal_sample)
        self.trainer.train_step(states, actions, rewards, next_states, is_game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 150 - self.numer_of_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_avg_scores = []
    total_score = 0
    record_score = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = game.get_state()
        action_final_move = agent.get_action(state_old)
        reward, is_game_over, score = game.play_step(action_final_move)
        state_new = game.get_state()

        agent.train_short_memory(state_old, action_final_move, reward, state_new, is_game_over)
        agent.remember(state_old, action_final_move, reward, state_new, is_game_over)

        if is_game_over:
            game.reset()
            agent.numer_of_games += 1
            agent.train_long_memory()

            if score > record_score:
                record_score = score
                agent.model.save()

            print('Game: ', agent.numer_of_games, 'Score: ', score, 'Record: ', record_score)
            plot_scores.append(score)
            total_score += score
            avg_score = total_score / agent.numer_of_games
            plot_avg_scores.append(avg_score)
            plot(plot_scores, plot_avg_scores)


if __name__ == '__main__':
    train()
