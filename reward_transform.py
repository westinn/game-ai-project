import numpy as np

def transform_reward(reward):
    return np.sign(reward)