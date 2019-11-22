import numpy as np
import pandas as pd

def assign_constant_points(projects, default_task=10):
    '''
    Takes in parsed project tree, with one level of tasks
    Outputs project tree with constant points assigned
    '''
    for goal in projects:
        for child in goal["ch"]:
            child["reward"] = default_task
    return projects

def assign_random_points(projects, distribution_fxn = np.random.normal, mean = 10, std = 2, *args):
    '''
    Takes in parsed project tree, with one level of tasks
    Outputs project tree with random points assigned according to distribution function with inputted args
    '''
    for goal in projects:
        for child in goal["ch"]:
            child["reward"] = distribution_fxn(mean, std)
    return projects
    