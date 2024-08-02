#!/usr/bin/env python3
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from patient_routing_env import PatientRoutingEnv
import numpy as np
import matplotlib.pyplot as plt


def test_environment():
    # Initialize the environment
    env = DummyVecEnv([lambda: PatientRoutingEnv(size=10)])

    # Load the trained model
    model = DQN.load("dqn_patient_routing_model")

    # Reset the environment
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    max_steps = 1000  # Adjust based on your environment and needs

    # Store results for visualization
    rewards = []
    observations = []

    # Run the agent in the environment
    while not done and step < max_steps:
        # Predict the action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1

        # Store results
        rewards.append(reward[0])
        observations.append(obs[0])

        # Optionally render the environment (if using human or rgb_array mode)
        if env.render_mode == "human":
            env.render()

    # Print results
    print(f"Total Reward: {total_reward}")
    print(f"Steps Taken: {step}")
    print(f"Done: {done}")

    # Close the environment
    env.close()

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Reward per Step')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Over Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    observations = np.array(observations)
    plt.plot(observations[:, 0], label='Agent X Position')
    plt.plot(observations[:, 1], label='Agent Y Position')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.title('Agent Position Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_environment()
