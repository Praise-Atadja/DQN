import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from patient_routing_env import PatientRoutingEnv
import numpy as np


def main():
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Initialize the environment with no rendering
    env = DummyVecEnv([lambda: PatientRoutingEnv(render_mode=None, size=5)])
    print("Environment initialized successfully.")

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
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        max_grad_norm=10,
    )

    # Training parameters
    total_episodes = 100  # Number of episodes to train
    total_timesteps_per_episode = 5000  # Increased timesteps per episode

    for episode in range(total_episodes):
        print(f"Starting training episode {episode + 1}...")
        model.learn(total_timesteps=total_timesteps_per_episode)
        print(f"Training episode {episode + 1} completed.")

    # Save the model
    model.save("dqn_patient_routing")

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
