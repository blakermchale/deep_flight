  
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
    INC_FORWARD_VEL = 0
    DEC_FORWARD_VEL = 1 
    YAW_LEFT = 2 
    YAW_RIGHT = 3

class AirSimEnv(gym.Env):

    GOAL_REWARD = 1000.
    COLLISION_REWARD = -1000.
    MAX_DIST_REWARD = 750.
    DIST_THRESH = 1.0

    def __init__(self):
        self.client = MyAirSimClient()
        self.goal = np.array([72.214,  -3.348, -2.])

        # Gym needs a defined object structure for observations and actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(84,84, 1))
        self.action_space = spaces.Discrete(4)
        
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
        velOffsetCmd, yaw_rate = self.interpret_action(action)

        self.client.simPause(False)
        self.client.modifyVel(velOffsetCmd, yaw_rate)

        observation = self.client.getDepthImage()
        reward, done, info = self.compute_reward()
        self.client.simPause(True)

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
        _, curr_pos = self.client.getState()

        info = {}
        done = False
        reward = 0
        distance = np.nan

        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.has_collided or (collision_info.time_stamp != 0)
        # print(collision_info.has_collided, collision_info.time_stamp)

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
                reward += self.MAX_DIST_REWARD * ((dist_from_home - distance) / dist_from_home)

        info["distance"] = distance
        info["has_collided"] = has_collided
        info["curr_pos"] = curr_pos
        return reward, done, info

    def interpret_action(self, action):
        """
        Interpret action enum and return back offset values for velocities and yaw rate.

        Args:
            action (Action): enum to chose what action to take

        Returns:
            offset (tuple(double, double)): vel offset, yaw rate
        """
        # Actions
        vel_offset = 1 # m/s
        yaw_rate = 15 # deg/s

        if action == Action.INC_FORWARD_VEL:
            offset = (vel_offset, 0)
        elif action == Action.DEC_FORWARD_VEL:
            offset = (-vel_offset, 0)
        elif action == Action.YAW_RIGHT:
            offset = (0, yaw_rate)
        elif action == Action.YAW_LEFT:
            offset = (0, -yaw_rate)
        else:
            raise ValueError(f"Action does not exist: {action}")

        return offset