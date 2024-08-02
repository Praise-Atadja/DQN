#!/usr/bin/env python3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from patient_routing_env import PatientRoutingEnv


def test_model():
    # Initialize the environment
    env = DummyVecEnv([lambda: PatientRoutingEnv()])  # Wrap environment

    # Load the trained model
    model = DQN.load('dqn_patient_routing_model')

    # Run the agent in the environment
    obs = env.reset()
    done = False
    step = 0

    while not done:
        env.render()  # Display the environment
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        step += 1
        print(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}")

    print("Test complete.")


if __name__ == "__main__":
    test_model()
