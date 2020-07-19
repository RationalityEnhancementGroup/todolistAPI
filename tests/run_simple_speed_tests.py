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


def run(test_mode: str, n_bins:list, n_goals: list, n_tasks: list,
        time_ests: list, today_hours: list, n_trials=1):
    
    API_ENDPOINT = f"{SERVER}/api/{test_mode}"
    
    time_results = dict()

    date = datetime.now()
    timestamp = f"{date.year:04d}_{date.month:02d}_{date.day:02d}_" \
                f"{date.hour:02d}_{date.minute:02d}_{date.second:02d}"

    main_df = pd.DataFrame()

    for nb in n_bins:
        
        for ng in n_goals:
            
            for nt in n_tasks:
                
                for te in time_ests:
                    
                    for h in today_hours:
                        
                        tm = h * 60

                        time_results[nt] = list()

                        print(
                            f"Number of bins: {nb} | "
                            f"Number of goals: {ng} | "
                            f"Number of tasks per goal: {nt} | "
                            f"Time estimate: {te} | "
                            f"Today minutes: {tm}"
                        )

                        data = {
                            "choice_mode":            "max",
                            "gamma":                  1e-9,
                            "loss_rate":              0,
                            "planning_fallacy_const": 1,
                            "slack_reward":           0,
                            "unit_penalty":           0,
            
                            "n_bins":                 nb,
                            "n_goals":                ng,
                            "n_tasks":                nt,
                            "time_est":               te,
                            "today_minutes":          tm,
            
                            "scale_type":             "no_scaling",
                            "scale_min":              0,
                            "scale_max":              0
                        }

                        n_timeouts = 0

                        for trial in tqdm(range(n_trials)):
                            start_time = time.time()
                            r = requests.post(url=API_ENDPOINT,
                                              data=json.dumps(data))
                            end_time = time.time()
        
                            output = r.text

                            log_info = {
                                "n_goals": ng,
                                "n_tasks": nt,
                                "n_bins":  nb,
                                "time_est": te,
                                "today_minutes": tm,
                                "timeout": False,
                                "trial":   trial
                            }

                            if "Testing" in output[:30]:
                                json_output = json.loads(output)
    
                                json_output.update(json_output["timer"])
                                del json_output["timer"]
    
                                log_info.update(json_output)
    
                                if VERBOSE:
                                    pprint(json_output)
                                time_results[nt].append(end_time - start_time)

                            elif "Timeout!" in output[:10]:
    
                                log_info.update({
                                    "status": "Timeout!",
                                    "timeout": True
                                })
    
                                print('\nTimeout!')
                                n_timeouts += 1

                            # elif "The API has encountered an error" in output:
                            #     print(output)

                            else:
                                log_info.update({
                                    "status": output
                                })
    
                                print(output)

                            # Clean-up database
                            # DB.request_log.delete_many({"user_id": USER_ID})
                            # DB.trees.delete_many({"user_id": USER_ID})

                            main_df = main_df.append(log_info,
                                                     ignore_index=True)
                            main_df.to_csv(f"{OUTPUT_PATH}/"
                                           f"{timestamp}_{test_mode}_"
                                           f"{SERVER_ABBR}_main.csv")

                        # Print results
                        print(
                            # f"\nNumber of bins: {nb}\n"
                            # f"Number of goals: {ng}\n"
                            # f"Number of tasks per goal: {nt}\n"
                            # f"Total number of tasks: {ng * nt}\n"
                            f"\nAverage time: {np.mean(time_results[nt]):.4f} | "
                            f"Timeouts: {n_timeouts}\n"
                            f"Time results: {time_results[nt]}\n")

                        if n_trials == n_timeouts:
                            print("Breaking...")
                            break

    return main_df


if __name__ == '__main__':
    
    LOCAL = 1
    HEROKU = 2
    VERBOSE = False
    
    MODE = HEROKU
    
    ALGORITHM = "simple_smdp"
    
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
        URI = os.environ['MONGODB_URI']
        CONN = MongoClient(URI)
        DB = CONN.heroku_g6l4lr9d
        
        SERVER = f"https://aqueous-hollows-34193.herokuapp.com"
        SERVER_ABBR = "heroku"
    
    # Clean-up database
    # DB.request_log.delete_many({"user_id": USER_ID})
    # DB.trees.delete_many({"user_id": USER_ID})
    
    # Check number of entries in the DB
    # print("Request log collection count:",
    #       DB.request_log.find({"user_id": USER_ID}).count())
    # print("Trees collection count:",
    #       DB.trees.find({"user_id": USER_ID}).count())
    
    N_BINS = [
        # 1,
        2
    ]
    
    # N_GOALS = list(range(1, 11))
    N_GOALS = [
        1,
        2,
        3,
        4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # 10
    ]
    
    N_TASKS = [
        10,
        25,
        50,
        75,
        100,
        125,
        150,
        250,
        
        # 500, 750, 1000,
        # 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000,
        # 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000
    ]
    
    TIME_ESTS = [
        # 5,
        # 10,
        15,
        # 30,
        # 45,
        # 60,
    ]
    
    TODAY_HOURS = [
        8,
        12,
        16
    ]
    
    # Test SMDP for different number of goals and tasks
    run(
        n_bins=N_BINS,
        n_goals=N_GOALS,
        n_tasks=N_TASKS,
        n_trials=5,
        time_ests=TIME_ESTS,
        today_hours=TODAY_HOURS,
        
        # test_mode="averageSpeedTestSMDP",
        # test_mode="bestSpeedTestSMDP",
        # test_mode="exhaustiveSpeedTestSMDP",
        test_mode="realSpeedTestSMDP",
        # test_mode="worstSpeedTestSMDP",
    )
