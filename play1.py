#!/usr/bin/env python3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from patient_routing_env import PatientRoutingEnv
import time


def play_simulation():
    # Initialize the environment with no rendering initially
    env = DummyVecEnv([lambda: PatientRoutingEnv(size=5, render_mode=None)])

    # Load the trained model
    model = DQN.load("dqn_patient_routing_real")

    # Reset the environment
    obs = env.reset()
    done = False
    total_reward = 0
    max_steps = 100  # Increased steps to allow more exploration

    for step in range(max_steps):
        if done:
            break  # Stop if the goal is reached

        # Predict the action based on the current observation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        # Accumulate reward
        total_reward += reward[0]

        # Print progress for debugging
        print(f"Step: {step}, Obs: {obs}, Reward: {reward}, Done: {done}")

    # Check if the agent reached the goal
    if done:
        # Reinitialize the environment with rendering enabled
        env = DummyVecEnv(
            [lambda: PatientRoutingEnv(size=5, render_mode="human")])

        # Reset and re-run the simulation for visualization
        obs = env.reset()
        done = False
        for step in range(max_steps):
            if done:
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            env.render()
            # Adjust speed as needed
            time.sleep(1 / env.metadata["render_fps"])

        print("Successful simulation rendered.")
    else:
        print("The goal was not reached within the allowed steps.")

    # Print results
    print(f"Total Reward: {total_reward}")
    print(f"Goal Reached: {done}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    play_simulation()
