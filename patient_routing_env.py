#!/usr/bin/env python3
# patient_routing_env.py
import gym
from gym import spaces
import numpy as np


class PatientRoutingEnv(gym.Env):
    def __init__(self):
        super(PatientRoutingEnv, self).__init__()
        self.grid_size = 5
        # 4 actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(4,), dtype=int)
        self.agent_pos = [0, 0]
        self.goal_pos = [4, 4]
        self.obstacles = [[1, 1], [2, 2], [3, 3]]  # Obstacle positions

    def reset(self):
        self.agent_pos = [0, 0]
        return np.array(self.agent_pos + self.goal_pos)

    def step(self, action):
        if action == 0:  # Move up
            self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1:  # Move down
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        elif action == 2:  # Move left
            self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3:  # Move right
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)

        done = self.agent_pos == self.goal_pos
        reward = 100 if done else -100  # Simple reward function

        return np.array(self.agent_pos + self.goal_pos), reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[self.goal_pos[1], self.goal_pos[0]] = 2  # Goal
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = 1  # Obstacles
        grid[self.agent_pos[1], self.agent_pos[0]] = 3  # Agent

        if mode == 'rgb_array':
            # Convert grid to RGB for capturing
            img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            img[grid == 2] = [0, 255, 0]  # Goal: green
            img[grid == 1] = [0, 0, 255]  # Obstacles: blue
            img[grid == 3] = [255, 0, 0]  # Agent: red
            return img
        elif mode == 'human':
            print(grid)
        else:
            raise ValueError("Invalid action")


env = PatientRoutingEnv()
observation = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    env.render()
    if done:
        break