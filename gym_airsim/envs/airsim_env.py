  
import logging
import numpy as np
import random
from enum import IntEnum

import gym
from gym import spaces
from gym.utils import seeding

from gym_airsim.envs.MyAirSimClient import MyAirSimClient

logger = logging.getLogger(__name__)


class Action(IntEnum):
    NONE = 0
    INC_VEL_X = 1
    DEC_VEL_X = 2
    INC_VEL_Y = 3
    DEC_VEL_Y = 4
    INC_VEL_Z = 5
    DEC_VEL_Z = 6
    YAW_RIGHT = 7
    YAW_LEFT = 8


class AirSimEnv(gym.Env):

    GOAL_REWARD = 100.
    COLLISION_REWARD = -100.
    DIST_THRESH = 1.0

    def __init__(self):
        self.client = MyAirSimClient()
        self.goal = np.array([10., 10., -3.])

        # Gym needs a defined object structure for observations and actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84, 1))
        self.action_space = spaces.Discrete(9)
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        """
        Step forward in openai gym.

        Args:
            action (Action): enum to chose what action to take

        Returns:
            observation (object): agent's observation of the current environment
            reward (double): amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        info = {}
        velOffsetCmd = self.interpret_action(action)
        self.client.modifyVel(velOffsetCmd)
    
        observation = self.client.getDepthImage()
        reward, done, distance = self.compute_reward()

        info["distance"] = distance

        return observation, reward, done, info
        
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        self.client.sim_reset()
        return self.client.getDepthImage()

    def compute_reward(self):
        """
        Computes reward for current state.

        Returns:
            reward (float): reward for current state
            done (bool): True if the vehicle has collided or near the goal
        """
        has_collided, curr_pos = self.client.getState()

        done = False
        reward = 0
        distance = None
        if has_collided:
            reward = self.COLLISION_REWARD
            done = True
        else:
            distance = np.linalg.norm(self.goal - curr_pos)
            if distance < self.DIST_THRESH:
                reward = self.GOAL_REWARD
                done = True
            else:
                dist_from_home = np.linalg.norm(self.goal - self.client.home)
                reward += (self.GOAL_REWARD / 2.) * ((dist_from_home - distance) / dist_from_home)

        return reward, done, distance

    def interpret_action(self, action):
        """
        Interpret action enum and return back offset values for velocities and yaw rate.

        Args:
            action (Action): enum to chose what action to take

        Returns:
            offset (tuple(double, double, double, double)): x, y, z, yaw rate
        """
        # Actions
        vel_offset = 0.25 # m/s
        yaw_rate = 15 # deg/s

        if action == Action.NONE:
            offset = (0, 0, 0, 0)
        elif action == Action.INC_VEL_X:
            offset = (vel_offset, 0, 0, 0)
        elif action == Action.DEC_VEL_X:
            offset = (-vel_offset, 0, 0, 0)
        elif action == Action.INC_VEL_Y:
            offset = (0, vel_offset, 0, 0)
        elif action == Action.DEC_VEL_Y:
            offset = (0, -vel_offset, 0, 0)
        elif action == Action.INC_VEL_Z:
            offset = (0, 0, vel_offset, 0)
        elif action == Action.DEC_VEL_Z:
            offset = (0, 0, -vel_offset, 0)
        elif action == Action.YAW_RIGHT:
            offset = (0, 0, 0, yaw_rate)
        elif action == Action.YAW_LEFT:
            offset = (0, 0, 0, -yaw_rate)
        else:
            raise ValueError(f"Action does not exist: {action}")

        return offset       
