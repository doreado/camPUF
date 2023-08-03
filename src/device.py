import random
import numpy as np
import constants
import logging

def get_response_key(hf_noise, challenge):
    m = [hf_noise[i] for i in challenge]
    median = np.median(m)
    response = [1 if i >= median else 0 for i in m]

    return response
