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


def test_speed_mdp(n_goals, n_tasks, deadline_years, mixing_params, n_trials=1):
    time_results = dict()
    
    time_df = pd.DataFrame()
    tout_df = pd.DataFrame()

    break_flag = False

    for ng in n_goals:
        for nt in n_tasks:
            for dy in deadline_years:

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
                          "| Years to deadline:", dy,
                          "| Mixing parameter:", mp)

                    PARAMS = f"api/{ALGORITHM}/mdp/1/10/inf/0/inf/0/inf/{mp}" \
                             f"/0/0/tree/u123/getTasksForToday"
                    API_ENDPOINT = SERVER + PARAMS

                    for _ in tqdm(range(n_trials)):

                        start_time = time.time()
                        r = requests.post(url=API_ENDPOINT, data=json.dumps(data))
                        end_time = time.time()

                        output = r.text

                        print(output)

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
                          f"Number of years: {dy}\n"
                          f"Mixing parameter: {mp:.2f}\n"
                          f"Average time: {time_log[mp]:.4f}\n"
                          f"Timeouts: {tout_log[mp]}\n"
                          f"Time results: {time_results[mp]}\n")

                    # if n_trials == tout_log[mp]:
                    #     break_flag = True
                    #     print("Breaking...", end=" ")
                    #     break

                time_df[idx_name] = time_log
                tout_df[idx_name] = tout_log

                time_df.to_csv(f"{OUTPUT_PATH}/"
                               f"results_{SERVER_ABBR}_time.csv")
                tout_df.to_csv(f"{OUTPUT_PATH}/"
                               f"results_{SERVER_ABBR}_tout.csv")

                if break_flag:
                    break

            if break_flag:
                print("Done!\n")
                break

        # Reset break flag
        break_flag = False

    return time_df, tout_df


def test_speed_smdp(test_mode, n_bins, n_goals, n_tasks, n_trials=1):
    time_results = dict()

    date = datetime.now()
    timestamp = f"{date.year:04d}_{date.month:02d}_{date.day:02d}_" \
                f"{date.hour:02d}_{date.minute:02d}_{date.second:02d}"
    
    time_df = pd.DataFrame()
    tout_df = pd.DataFrame()
    
    break_flag = False
    
    main_df = pd.DataFrame()

    for nb in n_bins:

        for ng in n_goals:
            
            time_log = pd.Series(index=n_tasks)
            tout_log = pd.Series(index=n_tasks, dtype=np.int)
            
            print(os.getcwd())
        
            for nt in n_tasks:
                
                idx_name = f"{nb}_bins_{ng}_goals_{nt}_tasks" + "_1_years"
    
                with open(f"{PATH_NAME}/{idx_name}.json", "r") as file:
                    data = json.load(file)
    
                    time_results[nt] = list()
                    tout_log[nt] = 0
    
                    print(f"Number of bins: {nb} | "
                          f"Number of goals: {ng} | "
                          f"Number of tasks per goal: {nt}"
                    )
                    
                    PARAMS = f"api/smdp/mdp/30/14/inf/0/inf/0/inf/false/2/" \
                             f"max/{1-1e-9}/1/{nb}/1.39/0/0/0/1/0/1/" \
                             f"tree/u123/{test_mode}"
                    API_ENDPOINT = SERVER + PARAMS
    
                    for trial in tqdm(range(n_trials)):
    
                        start_time = time.time()
                        r = requests.post(url=API_ENDPOINT, data=json.dumps(data))
                        end_time = time.time()
    
                        output = r.text
                        
                        log_info = {
                            "n_goals": ng,
                            "n_tasks": nt,
                            "n_bins": nb,
                            "trial": trial
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
                                "status": "Timeout!"
                            })
                            
                            print('\nTimeout!')
                            tout_log[nt] += 1
                            
                        # elif "The API has encountered an error" in output:
                        #     print(output)
                        
                        else:
                            log_info.update({
                                "status": output
                            })
        
                            print(output)
    
                        # Clean-up database
                        DB.request_log.delete_many({"user_id": USER_ID})
                        DB.trees.delete_many({"user_id": USER_ID})
                        
                        main_df = main_df.append(log_info, ignore_index=True)
                        main_df.to_csv(f"{OUTPUT_PATH}/"
                                       f"{timestamp}_{test_mode}_"
                                       f"{SERVER_ABBR}_main.csv")
    
                    # Compute mean time
                    time_log[nt] = np.mean(time_results[nt])
                    
                    # Print results
                    print(f"\nNumber of bins: {nb}\n"
                          f"Number of goals: {ng}\n"
                          f"Number of tasks per goal: {nt}\n"
                          f"Total number of tasks: {ng * nt}\n"
                          f"Average time: {time_log[nt]:.4f}\n"
                          f"Timeouts: {tout_log[nt]}\n"
                          f"Time results: {time_results[nt]}\n")
    
                    if n_trials == tout_log[nt]:
                        break_flag = True
                        print("Breaking...", end=" ")
                        # break
    
                time_df[ng] = time_log
                tout_df[ng] = tout_log
    
                time_df.to_csv(f"{OUTPUT_PATH}/"
                               f"{timestamp}_{test_mode}_{SERVER_ABBR}_time.csv")
                tout_df.to_csv(f"{OUTPUT_PATH}/"
                               f"{timestamp}_{test_mode}_{SERVER_ABBR}_tout.csv")
    
                if break_flag:
                    print("Done!\n")
                    break
    
            # Reset break flag
            break_flag = False
    
    return main_df, time_df, tout_df


if __name__ == '__main__':
    
    LOCAL = 1
    HEROKU = 2
    VERBOSE = False
    
    MODE = HEROKU
    
    # ALGORITHM = "dp"
    ALGORITHM = "smdp"
    
    TEST_TYPE = "smdp"
    USER_ID = "__test__"
    
    PATH_NAME = f"data/{TEST_TYPE}"
    OUTPUT_PATH = f"output/{ALGORITHM}/{TEST_TYPE}"
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
    
    if MODE == HEROKU:
        # URI = os.environ['MONGODB_URI']
        URI = "mongodb://hC2P81mItQ16:a1R9ydF01dih@ds341557.mlab.com:41557/heroku_g6l4lr9d?retryWrites=false"
        CONN = MongoClient(URI)
        DB = CONN.heroku_g6l4lr9d
        
        SERVER = f"https://aqueous-hollows-34193.herokuapp.com/"
        SERVER_ABBR = "heroku"

    # Clean-up database
    DB.request_log.delete_many({"user_id": USER_ID})
    DB.trees.delete_many({"user_id": USER_ID})

    # Check number of entries in the DB
    print("Request log collection count:",
          DB.request_log.find({"user_id": USER_ID}).count())
    print("Trees collection count:",
          DB.trees.find({"user_id": USER_ID}).count())

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
        5,
        6,
        7,
        8,
        9,
        10
    ]
    
    N_TASKS = [
        25,
        50,
        75,
        100,
        125, 150, 250, 500, 750, 1000,
        1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000,
        # 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000
    ]
    
    # Test DP for different number of goals, tasks and mixing-parameter values
    # test_speed_mdp(
    #     n_goals=[10],
    #     n_tasks=[2500, 7500, 10000],  # 10, 50, 100, 250, 500, 750, 1000,
    #     deadline_years=[None],
    #     mixing_params=[0.00, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99],
    #     n_trials=3
    # )

    # Test SMDP for different number of goals and tasks
    test_speed_smdp(
        n_bins=N_BINS,
        n_goals=N_GOALS,
        n_tasks=N_TASKS,
        n_trials=5,
        test_mode="averageSpeedTestSMDP",
        # test_mode="bestSpeedTestSMDP",
        # test_mode="worstSpeedTestSMDP",
    )
