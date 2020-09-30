import json
import numpy as np
import os
import pandas as pd
import requests
import time

from datetime import datetime
from pprint import pprint
from pymongo import MongoClient
from tqdm import tqdm


def run(n_trials=1, save=True):
    
    time_results = dict()
    
    date = datetime.now()
    timestamp = f"{date.year:04d}_{date.month:02d}_{date.day:02d}_" \
                f"{date.hour:02d}_{date.minute:02d}_{date.second:02d}"
    
    main_df = pd.DataFrame()

    for nb in N_BINS:
        for ng in N_GOALS:
            for bf in BRANCHING_FACTORS:
                for d in DEPTHS:
                    
                    # Read JSON file
                    with open(f"data/smdp/{ng}g_{bf}bf_{d}d.json") as json_file:
                        file = json.load(json_file)
                        
                    # Compute number of tasks
                    nt = bf ** d
                    
                    # Generate URL parameters
                    params = '/'.join([
                        '60',  # default_time_est
                        '14',  # default_deadline
                        'inf',  # allowed_task_time
                        '0',  # min_sum_of_goal_values
                        'inf',  # max_sum_of_goal_values
                        '0',  # min_goal_value_per_goal_duration
                        'inf',  # max_goal_value_per_goal_duration
                        'false',  # points_per_hour
                        '2',  # rounding
                        'max',  # choice_mode
                        '0.999999',  # gamma
                        '0',  # loss_rate
                        str(nb),  # num_bins
                        PLANNING_FALLACY_CONST,  # planning_fallacy_const
                        '0.0001',  # slack_reward
                        '0',  # unit_penalty
                        # 'min_max',          # scale_type
                        # '1',                # scale_min
                        # '2',                # scale_max
                        'tree',  # dummy parameter
                        '__test__',  # userID
                        'speedTest'  # function name
                    ])
                    
                    # Set time frame in the JSON dict
                    file["sub_goal_min_time"] = SUB_GOAL_MIN_TIME
                    file["sub_goal_max_time"] = SUB_GOAL_MAX_TIME

                    # Generate complete URL
                    API_ENDPOINT = f"{SERVER}/api/smdp/mdp/{params}"

                    print(
                        f"URL: {API_ENDPOINT}\n"
                        f"- Number of bins: {nb}\n"
                        f"- Number of goals: {ng}\n"
                        f"- Branching factor: {bf}\n"
                        f"- Depth: {d}\n"
                        f"- Number of tasks per goal: {nt}\n"
                        f"- Total number of tasks: {ng * nt}"
                    )
                    
                    # Initialize timemout counter
                    n_timeouts = 0
                    
                    # Initialize list of time results
                    time_results[nt] = list()

                    for trial in tqdm(range(n_trials)):
                        
                        # Send request and receive output
                        tic = time.time()
                        
                        output = requests.post(
                            url=API_ENDPOINT, data=json.dumps(file)).text
                        
                        toc = time.time()
                        
                        # Initialize log
                        log_info = {
                            "branching_factor": bf,
                            "depth":            d,
                            "n_bins":           nb,
                            "n_goals":          ng,
                            "n_tasks":          nt,
                            "n_tasks_total":    ng * nt,
                            "status":           "",
                            "time_frame":       SUB_GOAL_MAX_TIME,
                            "timeout":          False,
                            "trial":            trial
                        }
                        
                        # If the request was successful
                        if '{"status": "The procedure took' in output[:30]:
                            
                            # Convert output to JSON
                            json_output = json.loads(output)["timer"]
                            
                            # Update log with timing outputs
                            log_info.update(json_output)
                            
                            if VERBOSE:
                                pprint(json_output)
                            
                            # Add execution time
                            time_results[nt].append(toc - tic)
                        
                        # If timeout occurred
                        elif "Timeout!" in output[:10]:
                            
                            log_info.update({
                                "status":  "Timeout!",
                                "timeout": True
                            })
                            
                            print('\nTimeout!')
                            n_timeouts += 1
                        
                        # elif "The API has encountered an error" in output:
                        #     print(output)
                        
                        # If something unusual happened
                        else:
                            log_info.update({
                                "status": output
                            })
                            
                            print(output)
                            n_timeouts += 1

                        # Clean-up database
                        DB.request_log.delete_many({"api_method": "speedTest"})
                        DB.trees.delete_many({"api_method": "speedTest"})
                        
                        main_df = main_df.append(log_info,
                                                 ignore_index=True)
                        
                        if save:
                            main_df.to_csv(f"{OUTPUT_PATH}/{timestamp}_"
                                           f"{SERVER_ABBR}_{SUB_GOAL_MAX_TIME}.csv")
                    
                    # Print results
                    print(
                        f"\n"
                        f"- Average time: {np.mean(time_results[nt]):.4f}\n"
                        f"- Timeouts: {n_timeouts}\n"
                        f"- Time results: {time_results[nt]}\n")
                    
                    if n_trials == n_timeouts:
                        print("Breaking...")
                        break
    
    return main_df


if __name__ == '__main__':
    
    LOCAL = 1
    HEROKU = 2
    VERBOSE = False
    
    MODE = LOCAL
    
    ALGORITHM = "multilevel_smdp"
    
    TEST_TYPE = "smdp"
    USER_ID = "__test__"
    
    OUTPUT_PATH = f"output/{ALGORITHM}/{TEST_TYPE}"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    DB = None
    URI = None
    
    SERVER = None
    SERVER_ABBR = None
    
    if MODE == LOCAL:
        SERVER = f"http://127.0.0.1:6789"
        SERVER_ABBR = "local"
        
        URI = "mongodb://ai4productivity:ai4productivity@127.0.0.1/ai4productivity"
        CONN = MongoClient(URI)
        DB = CONN["ai4productivity"]
    
    if MODE == HEROKU:
        # URI = os.environ['MONGODB_URI']
        URI = "mongodb://hC2P81mItQ16:a1R9ydF01dih@ds341557.mlab.com:41557/heroku_g6l4lr9d?retryWrites=false"
        CONN = MongoClient(URI)
        DB = CONN.heroku_g6l4lr9d
        
        SERVER = f"https://aqueous-hollows-34193.herokuapp.com"
        SERVER_ABBR = "heroku"
    
    # Clean-up database
    # DB.request_log.delete_many({"api_method": "speedTest"})
    # DB.trees.delete_many({"api_method": "speedTest"})
    #
    # # Check number of entries in the DB
    # print("Request log collection count:",
    #       DB.request_log.find({"user_id": USER_ID}).count())
    # print("Trees collection count:",
    #       DB.trees.find({"user_id": USER_ID}).count())

    BIAS = None
    SCALE = None

    N_BINS = [
        1,
        2
    ]

    # Branching factor
    BRANCHING_FACTORS = [
        1,
        2,
        3,
        4,
        5,
        6
    ]

    # Tree depths
    DEPTHS = [
        1,
        2,
        3,
        4,
        5,
        6,
        # 7
    ]

    # Number of goals
    N_GOALS = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10
    ]

    PLANNING_FALLACY_CONST = '15'
    
    SUB_GOAL_MIN_TIME = '0'  # No lower time restriction

    # SUB_GOAL_MAX_TIME = '0'    # All tasks
    SUB_GOAL_MAX_TIME = 'inf'  # All first-level sub-goals

    # Test SMDP for different number of goals and tasks
    run(n_trials=5, save=True)
