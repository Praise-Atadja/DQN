#!/usr/bin/env python3
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from patient_routing_env import PatientRoutingEnv
import time


def play_simulation():
    # Initialize the environment
    env = DummyVecEnv([lambda: PatientRoutingEnv(size=5, render_mode="human")])

    # Load the trained model
    model = DQN.load("dqn_patient_routing")

    # Reset the environment
    obs = env.reset()
    done = False
    total_reward = 0
    start_time = time.time()
    max_time = 300  # 5 minutes in seconds

    while (time.time() - start_time) < max_time:
        if done:
            # Restart the environment if done
            obs = env.reset()
            done = False

        # Predict the action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        # Accumulate reward
        total_reward += reward[0]

        # Render the environment
        env.render()

        # Delay to ensure smooth animation
        time.sleep(1 / env.metadata["render_fps"])

    # Print results
    print(f"Total Reward: {total_reward}")
    print(f"Simulation Done: {done}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    play_simulation()
