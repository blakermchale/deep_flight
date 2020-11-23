import gym
import numpy as np
import random
from collections import deque
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from gym_airsim.envs.airsim_env import Action


class DQNAgent:
    # TODO: add in load model functionallity to save/load successful previous models and test
    def __init__(self, env, max_mem=100000, epsilon=0.95, gamma=0.9, epsilon_decay=0.995, 
                 learning_rate=0.005, tau=0.125, batch_size=32):
        self.env = env
        self.memory = deque(maxlen=max_mem)

        self.epsilon = epsilon # explore prob
        self.epsilon_min = 0.01 
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma # discount factor
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        """
        Creates TF model that takes in observation as input and predicts Q-values.
        """
        model = Sequential()
        obs_shape = self.env.observation_space.shape
        # input layer
        model.add(Conv2D(16, (8, 8), strides=(4, 4), input_shape=obs_shape, activation="relu"))
        # hidden layers
        model.add(Conv2D(32, (4, 4), strides=(2, 2), activation="relu"))
        model.add(Conv2D(32, (3, 3), strides=(1, 1), activation="relu"))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        # output layer
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model

    def act(self, obs):
        """
        Chose to take an action which is either random or from the model.
        Based on epsilon (explor probability)
        
        Args:
            obs (object): current observation of environment

        Returns:
            action (Action): action to take
        """
        # perform epsilon decay
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        # explore vs exploit
        if np.random.random() < self.epsilon:
            return Action(self.env.action_space.sample())

        # print("obs shape %s\n" % str(obs.shape))
        return Action(np.argmax(self.model.predict(obs)[0]))

    def remember(self, obs, action, reward, next_obs, done):
        """
        Create history of previous actions and their affects on states.

        Args:
            obs (object): current observation of environment
            action (Action): the action that was taken
            reward (float): amount of reward after action
            next_obs (object): observation of environment after action
            done (bool): whether the episode has ended
        """
        self.memory.append([obs, action, reward, next_obs, done])

    def replay(self):
        """
        Replay random previous examples from memory and update the model with a new associated
        Q value.
        """
        # wait until enough actions have been performed to train the model
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        obs_lst = []
        target_lst = []
        for sample in samples:
            obs, action, reward, next_obs, done = sample
            # obs_lst.append(obs)
            
            # print("obs shape (%f,%f)\n" % (obs.shape[0], obs.shape[1]))
            # print("next_obs shape (%f,%f)\n" % (next_obs.shape[0], next_obs.shape[1]))

            # write to png 
            # import os
            # import airsim
            # airsim.write_png(os.path.normpath('image.png'), obs.reshape(84,84)) 

            # start = time.time()
            target = self.target_model.predict(obs)
            # predict_dt = time.time() - start
            if done:
                target[0][action.value] = reward
            else:
                # bellman update equation
                next_Q = max(self.target_model.predict(next_obs)[0])
                target[0][action.value] = reward + next_Q * self.gamma
            # start = time.time()
            self.model.fit(obs, target, epochs=1, verbose=0)
            # fit_dt = time.time() - start
            # print(f"Predict DT: {predict_dt:.2f}, Fit DT: {fit_dt:.2f}")
            # target_lst.append(target)
        # start = time.time()
        # self.model.train_on_batch(obs_lst, target_lst)
        # fit_dt = time.time() - start
        # print(f"Fit DT: {fit_dt:.2f}")


    def target_train(self):
        """
        Update the weights of the target model. The target model is essentially a more stable
        version of the main model. This update is weighted by parameter tau.
        """
        
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        
        # scale target weights with model weights and tau
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1-self.tau)        
        
        self.target_model.set_weights(target_weights)

    def save_model(self, path):
        """
        Save the main model weights to a given path.

        Inputs:
            path (string): path to the saved model file
        """
        self.model.save(path)


def main():
    # allow for GPU on windows
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Constants
    GAMMA   = 0.9
    EPSILON = .95
    EPSILON_DECAY = 0.995
    TAU = 0.125
    MAX_MEM = 100000
    LEARNING_RATE = 0.005

    BATCH_SIZE = 32
    
    EPISODES  = 1000
    STEPS = 500

    OUT_DIR = "training_results/"

    # Create gym environment
    env = gym.make("gym_airsim:airsim-v0")
    env = gym.wrappers.Monitor(env, OUT_DIR, force=True)
     # TODO: Is this needed?
     #  Currently needed since gym monitor wrapper requires that an env reach a done state to reset
    STEPS = env.spec.max_episode_steps - 1 

    # Set goal
    env.goal = np.array([10., 0., -3.])

    # Create deep q-learning agent
    dqn_agent = DQNAgent(env=env, max_mem=MAX_MEM, gamma=GAMMA, epsilon=EPSILON, tau=TAU,
                         batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

    # Run episodes
    for episode in range(EPISODES):
        # Reset environment and variables
        print(f"Episode: {episode}")
        step = 0
        obs = env.reset()

        # Loop until episode is done or it reached the max # of steps
        while True:
            # perform action
            action = dqn_agent.act(obs)
            next_obs, reward, done, info = env.step(action)

            # fit model with new actions
            start = time.time()
            dqn_agent.remember(obs, action, reward, next_obs, done)
            rem_dt = time.time() - start
            start = time.time()
            dqn_agent.replay()
            rep_dt = time.time() - start
            start = time.time()
            dqn_agent.target_train()
            targ_dt = time.time() - start

            obs = next_obs
            if step >= STEPS:
                print(f"\nReached max # of steps")
                break
            if done:
                print(f"\nDone")
                break
            
            has_collided, curr_pos = env.client.getState()
            distance = info["distance"]
            print(f"Step: {step:02d} Action: {action.name:10} Reward: {reward:+.2f} "
                  f"Position: ({curr_pos[0]:+.2f}, {curr_pos[1]:+.2f}, {curr_pos[2]:+.2f}) "
                  f"Distance: {distance:+.2f} "
                  f"Collided: {has_collided} REM DT: {rem_dt:.2f} REP DT: {rep_dt:.2f} TARG DT: {targ_dt:.2f}", end='\r')
            step += 1

        if episode % 50 == 0:
            dqn_agent.save_model("ep-{}.model".format(episode))


if __name__ == "__main__":
    main()
