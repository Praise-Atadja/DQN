#!/usr/bin/env python3
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from patient_routing_env import PatientRoutingEnv


class CustomDQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(CustomDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),  # First hidden layer
            nn.ReLU(),
            nn.Linear(64, 64),  # Second hidden layer
            nn.ReLU(),
            nn.Linear(64, n_actions)  # Output layer with number of actions
        )

    def forward(self, x):
        return self.net(x)


def train_model():
    # Create the environment
    env = DummyVecEnv([lambda: PatientRoutingEnv()])  # Wrap environment

    # Define the custom network
    env_sample = env.reset()
    # Number of features in the observation space
    obs_dim = env_sample.shape[1]
    n_actions = env.action_space.n  # Number of actions

    # Initialize the model with custom policy
    policy_kwargs = dict(
        net_arch=[64, 64]
    )
    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # Train the model
    model.learn(total_timesteps=50000)

    # Save the trained model
    model.save('dqn_patient_routing_model')
    print("Model trained and saved.")


if __name__ == "__main__":
    train_model()
