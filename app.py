#!/usr/bin/env python3
import streamlit as st
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from patient_routing_env import PatientRoutingEnv
import time


def render_grid(env):
    grid_size = env.envs[0].grid_size
    grid = np.zeros((grid_size, grid_size), dtype=int)
    goal_pos = env.envs[0].goal_pos
    agent_pos = env.envs[0].agent_pos
    obstacles = env.envs[0].obstacles

    grid[goal_pos[1], goal_pos[0]] = 2  # Goal
    for obs in obstacles:
        grid[obs[1], obs[0]] = 1  # Obstacles
    grid[agent_pos[1], agent_pos[0]] = 3  # Agent

    # Convert grid to RGB
    img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    img[grid == 2] = [0, 255, 0]  # Goal: green
    img[grid == 1] = [0, 0, 255]  # Obstacles: blue
    img[grid == 3] = [255, 0, 0]  # Agent: red
    return img


def play_simulation():
    # Initialize the environment
    env = DummyVecEnv([lambda: PatientRoutingEnv()])  # Wrap environment

    # Load the trained model
    model = DQN.load('dqn_patient_routing_model')

    # Streamlit setup
    st.title('Agent Simulation')
    stframe = st.empty()

    # Run the agent in the environment
    obs = env.reset()
    done = False
    step = 0
    max_steps = 5 * 60 * 20  # 5 minutes at 20 fps

    while not done and step < max_steps:
        # Predict the action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        # Render the environment
        img = render_grid(env)
        stframe.image(
            img, caption=f'Step: {step}', use_column_width=True, channels='RGB')

        # Increment step counter
        step += 1

        # Pause to control the speed of the simulation
        time.sleep(0.05)

    st.write("Simulation complete.")


if __name__ == "__main__":
    play_simulation()
