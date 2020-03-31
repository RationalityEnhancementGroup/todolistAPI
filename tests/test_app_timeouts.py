#%%
import json
import numpy as np
import os
import pandas as pd
import requests
import time

from pprint import pprint
from pymongo import MongoClient
from tqdm import tqdm

#%% sending post request and saving response as response object

LOCAL = 1
HEROKU_STAGING = 2
VERBOSE = True

MODE = HEROKU_STAGING

TEST_TYPE = "no_deadlines"
USER_ID = "__test__"

PATH_NAME = f"tests/data/{TEST_TYPE}"
OUTPUT_PATH = f"tests/output/{TEST_TYPE}"
os.makedirs(OUTPUT_PATH, exist_ok=True)

DB = None
URI = None

SERVER = None
SERVER_ABBR = None

if MODE == LOCAL:
    SERVER = f"http://127.0.0.1:6789/"
    SERVER_ABBR = "local"

    URI = "mongodb://ai4productivity:ai4productivity@127.0.0.1/ai4productivity"
    CONN = MongoClient(URI)
    DB = CONN["ai4productivity"]

if MODE == HEROKU_STAGING:
    URI = os.environ['MONGODB_URI']
    CONN = MongoClient(URI)
    DB = CONN.heroku_g6l4lr9d

    SERVER = f"https://aqueous-hollows-34193.herokuapp.com/"
    SERVER_ABBR = "heroku_staging"

#%% Check number of entries in the DB

# Clean-up database
DB.request_log.delete_many({"user_id": USER_ID})
DB.trees.delete_many({"user_id": USER_ID})

print("Request log collection count:",
      DB.request_log.find({"user_id": USER_ID}).count())
print("Trees collection count:",
      DB.trees.find({"user_id": USER_ID}).count())


#%%
def test_size(n_goals, n_tasks, mixing_params, n_trials=1):
    time_results = dict()
    
    time_df = pd.DataFrame()
    tout_df = pd.DataFrame()
    
    for ng in n_goals:
        for nt in n_tasks:
            
            time_log = pd.Series(index=mixing_params)
            tout_log = pd.Series(index=mixing_params)
            
            idx_name = f"{ng}_goals_{nt}_tasks"
            
            with open(f"{PATH_NAME}/{idx_name}.json", "r") as file:
                data = json.load(file)
                
            for mp in mixing_params:
                time_results[mp] = list()
                tout_log[mp] = 0
                
                print("Number of goals:", ng,
                      "| Number of tasks per goal:", nt,
                      "| Mixing parameter:", mp)
        
                PARAMS = f"api/greedy/mdp/1/10/inf/0/inf/0/inf/{mp}" \
                         f"/0/0/tree/u123/getTasksForToday"
                API_ENDPOINT = SERVER + PARAMS
        
                for _ in tqdm(range(n_trials)):
                    
                    start_time = time.time()
                    r = requests.post(url=API_ENDPOINT, data=json.dumps(data))
                    end_time = time.time()
        
                    output = r.text
        
                    if output[0] == '"':
                        print('\nTimeout!')
                        tout_log[mp] += 1
                    elif output[0] == '<':
                        print(output)
                    else:
                        if VERBOSE:
                            pprint(json.loads(output))
                        time_results[mp].append(end_time - start_time)
                        
                    # Clean-up database
                    DB.request_log.delete_many({"user_id": USER_ID})
                    DB.trees.delete_many({"user_id": USER_ID})
        
                # Print results
                time_log[mp] = np.mean(time_results[mp])
                print(f"\nNumber of goals: {ng}\n"
                      f"Number of tasks per goal: {nt}\n"
                      f"Total number of tasks: {ng * nt}\n"
                      f"Mixing parameter: {mp:.2f}\n"
                      f"Average time: {time_log[mp]:.4f}\n"
                      f"Timeouts: {tout_log[mp]}\n"
                      f"Time results: {time_results[mp]}\n")
        
            time_df[idx_name] = time_log
            tout_df[idx_name] = tout_log
        
            time_df.to_csv(f"{OUTPUT_PATH}/"
                           f"results_{SERVER_ABBR}_{TEST_TYPE}_time.csv")
            tout_df.to_csv(f"{OUTPUT_PATH}/"
                           f"results_{SERVER_ABBR}_{TEST_TYPE}_tout.csv")

    return time_df, tout_df


#%%
if __name__ == '__main__':
    
    # Tests for different values of goals, tasks and mixing-parameter values
    test_size(
        n_goals=[10],
        n_tasks=[10, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000],
        mixing_params=[0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99],
        n_trials=10
    )
