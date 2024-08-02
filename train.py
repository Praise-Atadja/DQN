#!/usr/bin/env python3
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
# Adjust import according to your file structure
from patient_routing_env import PatientRoutingEnv


def main():
    # Initialize the environment
    env = DummyVecEnv([lambda: PatientRoutingEnv(size=5)])

    # Define the model
    model = DQN(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
    )

    # Train the model
    model.learn(total_timesteps=10000)

    # Save the model
    model.save("dqn_patient_routing_model")


if __name__ == "__main__":
    main()
