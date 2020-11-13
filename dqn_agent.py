import gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from gym_airsim.envs.airsim_env import Action


class DQNAgent:
    def __init__(self, env):
        self.env = env
        pass

    def create_model(self):
        pass

    def act(self):
        pass

    def remember(self):
        pass

    def replay(self):
        pass

    def target_train(self):
        pass

    def save_model(self):
        pass


def main():
    # Constants
    GAMMA   = 0.9
    EPSILON = .95

    EPISODES  = 1000
    STEPS = 500

    # Create gym environment
    env = gym.make("gym_airsim:airsim-v0")

    # Create deep q-learning agent
    dqn_agent = DQNAgent(env=env)

    # Run episodes
    for episode in range(EPISODES):
        # Reset environment and variables
        step = 0
        obs = env.reset()

        # Loop until episode is done or it reached the max # of steps
        while True:
            action = Action(env.action_space.sample())
            # action = Action.YAW_RIGHT
            obs, reward, done, _ = env.step(action)
            if step >= STEPS:
                print(f"\nReached max # of steps")
                break
            if done:
                print(f"\nDone")
                break
            print(f"Actions taken: {step:02d}, Action: {action.name}", end='\r')
            step += 1
    pass


if __name__ == "__main__":
    main()
