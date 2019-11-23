import numpy as np
import pandas as pd
import requests
import json
from src.utils import tree_to_old_api_json

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

def assign_old_api_points(projects, user_num=None):
    '''
    input: parsed project tree and user num
    output: #TODO
    '''

    #send post request
    old_json = tree_to_old_api_json(projects)
    request_string = "https://todo-gamification.herokuapp.com/todo/goals/{}/1".format(user_num) #TODO clearing right now, but maybe not most efficient
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(request_string, json = json.dumps(old_json), headers=headers)
    print(r)
    raise NotImplementedError

def assign_length_points(projects):
    '''
    Takes in parsed and flattened project tree
    Outputs project tree with points assigned according to length heuristic
    '''
    for goal in projects:
        value_per_minute = goal["value"]/float(sum([child["est"] for child in goal["ch"]]))
        for child in goal["ch"]:
            child["val"] = child["est"]/float(value_per_minute)
    return projects