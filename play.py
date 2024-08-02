#!/usr/bin/env python3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from patient_routing_env import PatientRoutingEnv
import time
import numpy as np


def render_grid(env):
    # Get the current state representation from the environment
    grid = np.zeros((env.envs[0].grid_size, env.envs[0].grid_size), dtype=int)
    grid[env.envs[0].goal_pos[1], env.envs[0].goal_pos[0]] = 2  # Goal
    for obs in env.envs[0].obstacles:
        grid[obs[1], obs[0]] = 1  # Obstacles
    grid[env.envs[0].agent_pos[1], env.envs[0].agent_pos[0]] = 3  # Agent

    # Clear the terminal screen
    print("\033c", end="")

    # Create a string representation of the grid
    grid_str = ""
    for row in grid:
        grid_str += ' '.join({
            0: '.',
            1: 'X',
            2: 'G',
            3: 'A'
        }.get(cell, ' ') for cell in row) + '\n'

    # Print the grid
    print(grid_str)


def play_simulation():
    # Initialize the environment
    env = DummyVecEnv([lambda: PatientRoutingEnv()])  # Wrap environment

    # Load the trained model
    model = DQN.load('dqn_patient_routing_model')

    # Run the agent in the environment
    obs = env.reset()
    done = False
    step = 0
    max_steps = 5 * 60 * 20  # 5 minutes at 20 fps

    while not done and step < max_steps:
        # Render the environment to the terminal
        render_grid(env)
        time.sleep(0.05)  # Delay to visualize movements

        # Predict the action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        step += 1

    print("Simulation complete.")


if __name__ == "__main__":
    play_simulation()
