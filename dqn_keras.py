import gym
import numpy as np
import random
from collections import deque
import time
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, Permute
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from gym_airsim.envs.airsim_env import Action


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='gym_airsim:airsim-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

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

WINDOW_LENGTH = 1

env = gym.make("gym_airsim:airsim-v0")

nb_actions = env.action_space.n

model = Sequential()
obs_shape = env.observation_space.shape

input_shape = (84, 84, 1)# (WINDOW_LENGTH,) + obs_shape
# input layer
# (width, height, channels)
# model.add(Permute((2, 3, 1), input_shape=input_shape))
# model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=input_shape))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (4, 4), strides=(2, 2)))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3), strides=(1, 1)))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))

model = Sequential()
obs_shape = env.observation_space.shape
# input layer
model.add(Conv2D(16, (8, 8), strides=(4, 4), input_shape=obs_shape, activation="relu"))
# hidden layers
model.add(Conv2D(32, (4, 4), strides=(2, 2), activation="relu"))
model.add(Conv2D(32, (3, 3), strides=(1, 1), activation="relu"))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
# output layer
model.add(Dense(env.action_space.n))
print(model.summary())


train = True

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)                        #reduce memmory


# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                              nb_steps=100000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, 
               enable_double_dqn=True, 
               enable_dueling_network=True, dueling_type='avg', 
               target_model_update=1e-2, policy=policy, gamma=.99)

dqn.compile(Adam(lr=0.00025), metrics=['mae'])


# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in tensorflow.keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
