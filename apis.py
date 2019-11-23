import numpy as np
import pandas as pd

def assign_constant_points(projects, default_task=10):
    '''
    Takes in parsed project tree, with one level of tasks
    Outputs project tree with constant points assigned
    '''
    for goal in projects:
        for child in goal["ch"]:
            child["val"] = default_task
    return projects

def assign_random_points(projects, distribution_fxn = np.random.normal, fxn_args = (10,2)):
    '''
    Takes in parsed project tree, with one level of tasks
    Outputs project tree with random points assigned according to distribution function with inputted args
    '''
    for goal in projects:
        for child in goal["ch"]:
            child["val"] = distribution_fxn(*fxn_args)
    return projects
    
def assign_hierarchical_points(projects):
    raise NotImplementedError

def assign_old_api_points(projects):
    raise NotImplementedError