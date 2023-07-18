import random
from collections import namedtuple
from enum import Enum

import numpy
import pygame

pygame.init()
font = pygame.font.Font('/Users/tietzeu/private/ai-projects/py-snake/assets/arial.ttf', 25)


# font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')
DirectionChanges = namedtuple('DirectionChanges', 'left, right')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN1 = (44, 84, 58)
GREEN2 = (134, 140, 114)
BLACK = (0, 0, 0)
BACKGROUND_SNAKE = (189, 182, 154)

BLOCK_SIZE = 20
SPEED = 1000


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.food = None
        self.score = None
        self.snake = None
        self.head = None
        self.direction = None
        self.deadlock = False
        self.is_deadlock_left = 0
        self.is_deadlock_right = 0
        self.frameIteration = None
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]

        self.score = 0
        self.food = None
        self.deadlock = False
        self._place_food()
        self.frameIteration = 0

    def _get_directions(self):
        direction_left = self.direction == Direction.LEFT
        direction_right = self.direction == Direction.RIGHT
        direction_up = self.direction == Direction.UP
        direction_down = self.direction == Direction.DOWN

        return [direction_left, direction_right, direction_up, direction_down]

    def _get_danger(self):
        head = self.head
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        direction_left, direction_right, direction_up, direction_down = self._get_directions()

        danger_straight = (direction_right and self.is_collision(point_right)) or \
                          (direction_left and self.is_collision(point_left)) or \
                          (direction_up and self.is_collision(point_up)) or \
                          (direction_down and self.is_collision(point_down))
        danger_right = (direction_up and self.is_collision(point_right)) or \
                       (direction_down and self.is_collision(point_left)) or \
                       (direction_left and self.is_collision(point_up)) or \
                       (direction_right and self.is_collision(point_down))
        danger_left = (direction_down and self.is_collision(point_right)) or \
                      (direction_up and self.is_collision(point_left)) or \
                      (direction_right and self.is_collision(point_up)) or \
                      (direction_left and self.is_collision(point_down))

        return [danger_straight, danger_right, danger_left]

    def get_state(self):
        direction_left, direction_right, direction_up, direction_down = self._get_directions()
        danger_straight, danger_right, danger_left = self._get_danger()

        state = [
            # Danger in next move direction
            danger_straight * 2,
            danger_right * 2,
            danger_left * 2,

            #  Move direction
            direction_left * 1,
            direction_right * 1,
            direction_up * 1,
            direction_down * 1,

            #  Food location
            1 * (self.food.x < self.head.x),  # food is left
            1 * (self.food.x > self.head.x),  # food is right
            1 * (self.food.y < self.head.y),  # food is up
            1 * (self.food.y > self.head.y),  # food is down
        ]

        # for row in range(0, int(self.h / BLOCK_SIZE)):
        #     for column in range(0, int(self.w / BLOCK_SIZE)):
        #         if Point(column * BLOCK_SIZE, row * BLOCK_SIZE) in self.snake[1:]:
        #             state.append(1)
        #         else:
        #             state.append(0)

        return numpy.array(state, dtype=int)

    def play_step(self, action):
        self.frameIteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Check for potential deadlock in direction
        danger_straight, danger_right, danger_left = self._get_danger()
        # direction_changes = self.count_direction_changes()
        # if self.deadlock is False:
        #     is_action_left = action is [0, 0, 1]
        #     is_action_right = action is [0, 1, 0]
        #     self.is_deadlock_left = is_action_left and danger_straight and direction_changes.left - direction_changes.right >= 3
        #     self.is_deadlock_right = is_action_right and danger_straight and direction_changes.right - direction_changes.left >= 3
        #     self.deadlock = self.is_deadlock_left or self.is_deadlock_right

        # 3. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 4. check if game over
        reward = 0
        game_over = False

        if self.deadlock is True:
            reward -= 30

        collision = self.is_collision()

        if collision or self.frameIteration > 100 * len(self.snake):
            if collision and collision == 2:
                reward -= 20
            elif collision == 1:
                reward -= 10
            elif self.frameIteration > 100 * len(self.snake):
                reward -= 10
            else:
                reward -= 10
            game_over = True
            return reward, game_over, self.score

        # 5. place new food or just move
        if self.head == self.food:
            reward += len(self.snake) * 10
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 6. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 7. return reward, game_over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return 1
        # hits itself
        if pt in self.snake[1:]:
            return 2

        return 0

    def calculate_board_size(self):
        return (self.w / BLOCK_SIZE) * (self.h / BLOCK_SIZE)

    def count_direction_changes(self, point_until=None):
        if point_until is None:
            point_until = self.snake[-1]

        left_turns = 0
        right_turns = 0

        for i in range(1, len(self.snake) - 1):
            if self.snake[i] is point_until:
                break

            point_before = self.snake[i - 1]
            point = self.snake[i]
            point_after = self.snake[i + 1]

            x1 = point_before.x
            x2 = point.x
            x3 = point_after.x
            y1 = point_before.y
            y2 = point.y
            y3 = point_after.y

            if x1 != x3 and y1 != y3:
                if x2 < x1 and x2 == x3 and y2 == y1 and y2 < y3:
                    right_turns += 1
                elif x2 == x1 and x2 < x3 and y2 < y1 and y2 == y3:
                    left_turns += 1
                elif x2 == x1 and x2 > x3 and y2 < y1 and y2 == y3:
                    right_turns += 1
                elif x2 > x1 and x2 == x3 and y2 == y1 and y2 < y3:
                    left_turns += 1
                elif x2 == x1 and x2 < x3 and y2 > y1 and y2 == y3:
                    right_turns += 1
                elif x2 < x1 and x2 == x3 and y2 == y1 and y2 > y3:
                    left_turns += 1
                elif x2 > x1 and x2 == x3 and y2 == y1 and y2 > y3:
                    right_turns += 1
                elif x2 == x1 and x2 > x3 and y2 > y1 and y2 == y3:
                    left_turns += 1

        return DirectionChanges(right=right_turns, left=left_turns)

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _update_ui(self):
        self.display.fill(BACKGROUND_SNAKE)

        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clockwise.index(self.direction)

        if numpy.array_equal(action, [1, 0, 0]):
            # no change
            self.direction = clockwise[index]
        elif numpy.array_equal(action, [0, 1, 0]):
            # right turn r -> d -> l -> u
            self.direction = clockwise[(index + 1) % 4]
        else:  # has to be [0,0,1]
            # left turn r -> u -> l -> d
            self.direction = clockwise[(index - 1) % 4]

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
