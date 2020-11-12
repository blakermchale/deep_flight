  
import logging
import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding

logger = logging.getLogger(__name__)


class AirSimEnv(gym.Env):
        
    def __init__(self):
        pass
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):

        return self.state, reward, done, info
        
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        """
        
        return self.state