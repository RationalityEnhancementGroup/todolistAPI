import numpy as np
import os
import pandas as pd
import sys
import time

from datetime import datetime
from math import factorial
from pprint import pprint
from tqdm import tqdm

from todolistMDP.smdp_utils import compute_pseudo_rewards, run_optimal_policy
from todolistMDP.smdp_test_generator import generate_test_case
from todolistMDP.to_do_list import Task, Goal, ToDoList


def print_item(item):
    for s in sorted(list(item.Q.keys())):
        for t in sorted(list(item.Q[s].keys())):
            for a in sorted(list(item.Q[s][t].keys()), key=lambda x: (x is None, x)):
                # for t_ in to_do_list.Q[s][t][a].keys():
                #     if t_ != "E":
                # print(s, t, a, t_, to_do_list.Q[s][t][a][t_])
                print(f"s: {s} | ", end="")
                print(f"t: {str(t) if t != np.PINF else 'inf':6s} | ", end="")
                print(f"a: {'-' if a is None else str(a):>3s} | ", end="")
                # print(f"t': {t_} | ", end="")
                print(f"Q: {item.Q[s][t][a]['E']:10.3f} | ", end="")
                print(f"f: {item.F[s][t][a]['E']:10.3f} | ", end="")
                print(f"r: {item.R[s][t][a]['E']:10.3f} | ", end="")
                print(f"r': {item.PR[s][t][a]['E']:10.3f} | ", end="")
                # print(f"Q: {item.Q[s][t][a]['E']} | ", end="")
                # print(f"f: {item.F[s][t][a]['E']} | ", end="")
                # print(f"r: {item.R[s][t][a]['E']} | ", end="")
                # print(f"r': {item.PR[s][t][a]['E']} | ", end="")
                print()
        print()


# ===== Constants =====
# LOSS_RATE = 0  # Default
LOSS_RATE = -1
# LOSS_RATE = -1e-3

# GAMMA = 0.9999  # Default
GAMMA = 0.9999
# GAMMA = 1 - 1e-3

N_BINS = 1
N_GOALS = 10
N_TASKS = 50
START_TIME = 0  # Default

TIME_SCALE = 1  # Default
# TIME_SCALE = 20

VALUE_SCALE = 1  # Default
# VALUE_SCALE = 10

PLANNING_FALLACY_CONST = 1  # Default
# PLANNING_FALLACY_CONST = 1.39

SLACK_REWARD = np.NINF
# SLACK_REWARD = 1
# SLACK_REWARD = 1e-1
# SLACK_REWARD = 1e-2
# SLACK_REWARD = 1e-3

# UNIT_PENALTY = 10
UNIT_PENALTY = .1
# UNIT_PENALTY = np.PINF

sys.setrecursionlimit(10000)

d_bm = [
    Goal(
        description="Goal A",
        # deadline=10,
        loss_rate=LOSS_RATE,
        num_bins=N_BINS,
        # penalty=-10,
        planning_fallacy_const=PLANNING_FALLACY_CONST,
        # reward=100,
        rewards={TIME_SCALE * 10: VALUE_SCALE * 100},
        tasks=[
             Task("Task A1", time_est=TIME_SCALE * 1),
             Task("Task A2", time_est=TIME_SCALE * 1)
        ],
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="Goal B",
        # deadline=10,
        loss_rate=LOSS_RATE,
        num_bins=N_BINS,
        # penalty=0,
        planning_fallacy_const=PLANNING_FALLACY_CONST,
        # reward=10,
        # rewards={TIME_SCALE * 10: 10},  # Simplified
        rewards={TIME_SCALE * 1: VALUE_SCALE * 10,
                 TIME_SCALE * 10: VALUE_SCALE * 10},
        tasks=[
             Task("Task B1", time_est=TIME_SCALE * 2),
             Task("Task B2", time_est=TIME_SCALE * 2)
        ],
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(description="Goal C",
         # deadline=6,
         loss_rate=LOSS_RATE,
         num_bins=N_BINS,
         # penalty=-1,
         planning_fallacy_const=PLANNING_FALLACY_CONST,
         # reward=100,
         # rewards={TIME_SCALE * 6: 100},  # Simplified
         rewards={TIME_SCALE * 1: VALUE_SCALE * 10,
                  TIME_SCALE * 6: VALUE_SCALE * 100},
         tasks=[
             Task("Task C1", time_est=TIME_SCALE * 3),
             Task("Task C2", time_est=TIME_SCALE * 3)
         ],
         unit_penalty=UNIT_PENALTY,
         ),
    Goal(description="Goal D",
         # deadline=40,
         loss_rate=LOSS_RATE,
         num_bins=N_BINS,
         # penalty=-10,
         planning_fallacy_const=PLANNING_FALLACY_CONST,
         # reward=10,
         # rewards={TIME_SCALE * 40: 10},  # Simplified
         rewards={TIME_SCALE * 20: VALUE_SCALE * 100,
                  TIME_SCALE * 40: VALUE_SCALE * 10},
         tasks=[
             Task("Task D1", time_est=TIME_SCALE * 3),
             Task("Task D2", time_est=TIME_SCALE * 3)
         ],
         unit_penalty=UNIT_PENALTY,
         ),
    Goal(
        description="Goal E",
        # deadline=70,
        loss_rate=LOSS_RATE,
        num_bins=N_BINS,
        # penalty=-110,
        planning_fallacy_const=PLANNING_FALLACY_CONST,
        # reward=10,
        # rewards={70: 10},  # Simplified
        rewards={TIME_SCALE * 60: VALUE_SCALE * 100,
                 TIME_SCALE * 70: VALUE_SCALE * 10},
        tasks=[
            Task("Task E1", time_est=TIME_SCALE * 3),
            Task("Task E2", time_est=TIME_SCALE * 3)
        ],
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="Goal F",
        # deadline=70,
        loss_rate=LOSS_RATE,
        num_bins=N_BINS,
        # penalty=-110,
        planning_fallacy_const=PLANNING_FALLACY_CONST,
        # reward=10,
        # rewards={70: 10},  # Simplified
        rewards={TIME_SCALE * 60: VALUE_SCALE * 100,
                 TIME_SCALE * 70: VALUE_SCALE * 10},
        tasks=[
            Task("Task F1", time_est=TIME_SCALE * 3),
            Task("Task F2", time_est=TIME_SCALE * 3)
        ],
        unit_penalty=UNIT_PENALTY,
    )
]

# goal = Goal(
#     description="Goal",
#     # deadline=100,
#     loss_rate=-1,
#     # reward=100,
#     rewards={100: 100},
#     penalty=-200,
#     tasks=[
#         Task("T1", deadline=10, time_est=2),
#         Task("T2", deadline=9,  time_est=1),
#         Task("T3", deadline=8,  time_est=4),
#         Task("T4", deadline=7,  time_est=3)
#     ]
# )

test_1 = [
    Goal(
        description="G1",
        loss_rate=LOSS_RATE,
        rewards={6: 1000},
        tasks=[
            Task("Task 1", time_est=1, deadline=6),
            Task("Task 2", time_est=2, deadline=5),
            Task("Task 3", time_est=3, deadline=4)
        ],
        unit_penalty=UNIT_PENALTY,
    ),
    # Goal(
    #     description="G2",
    #     loss_rate=LOSS_RATE,
    #     rewards={300: 500},
    #     tasks=[
    #         Task("Task 1", time_est=30, deadline=303),
    #         Task("Task 2", time_est=40, deadline=302),
    #         Task("Task 3", time_est=50, deadline=301)
    #     ],
    #     unit_penalty=UNIT_PENALTY,
    # ),
    # Goal(
    #     description="G3",
    #     loss_rate=LOSS_RATE,
    #     rewards={30: 30},
    #     tasks=[
    #         Task("Task 1", time_est=8, deadline=33),
    #         Task("Task 2", time_est=9, deadline=32),
    #         Task("Task 3", time_est=10, deadline=31)
    #     ],
    #     unit_penalty=UNIT_PENALTY,
    # )
]


test_2 = [
    Goal(
        description="G1",
        loss_rate=LOSS_RATE,
        planning_fallacy_const=PLANNING_FALLACY_CONST,
        rewards={2000: 5000},
        tasks=[
            Task("Task 1", time_est=100),
            Task("Task 2", time_est=200),
            Task("Task 3", time_est=300),
            Task("Task 4", time_est=400),
            Task("Task 5", time_est=500)
        ],
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="G2",
        loss_rate=LOSS_RATE,
        planning_fallacy_const=PLANNING_FALLACY_CONST,
        rewards={3000: 3000},
        tasks=[
            Task("Task 1", time_est=100),
            Task("Task 2", time_est=200),
            Task("Task 3", time_est=300),
            Task("Task 4", time_est=400),
            Task("Task 5", time_est=500)
        ],
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="G3",
        loss_rate=LOSS_RATE,
        planning_fallacy_const=PLANNING_FALLACY_CONST,
        rewards={4000: 10000},
        tasks=[
            Task("Task 1", time_est=100),
            Task("Task 2", time_est=200),
            Task("Task 3", time_est=300),
            Task("Task 4", time_est=400),
            Task("Task 5", time_est=500)
        ],
        unit_penalty=UNIT_PENALTY,
    )
]


single_goal = Goal(
    description="Single goal",
    loss_rate=LOSS_RATE,
    num_bins=N_BINS,
    planning_fallacy_const=PLANNING_FALLACY_CONST,
    rewards={100: 100},
    # tasks=[
    #     Task("Task 1", time_est=1, deadline=10),
    #     Task("Task 2", time_est=2, deadline=9),
    #     Task("Task 3", time_est=3, deadline=8),
    #     Task("Task 4", time_est=4, deadline=7),
    #     Task("Task 5", time_est=5, deadline=6),
    #     Task("Task 5", time_est=6, deadline=5),
    #     Task("Task 5", time_est=7, deadline=4),
    #     Task("Task 5", time_est=8, deadline=3),
    #     Task("Task 5", time_est=9, deadline=2),
    #     Task("Task 5", time_est=10, deadline=1)
    # ],
    # tasks=[
    #     Task("Task 1", time_est=1, deadline=6),
    #     Task("Task 2", time_est=1, deadline=3),
    #     Task("Task 3", time_est=2, deadline=5),
    #     Task("Task 4", time_est=2, deadline=2),
    # ],
    tasks=[
        Task("Task 1", time_est=1, deadline=6),
        Task("Task 2", time_est=2, deadline=5),
        Task("Task 3", time_est=3, deadline=3),
    ],
    unit_penalty=UNIT_PENALTY,
)

two_goals = [
    Goal(
        description="Goal 1",
        loss_rate=LOSS_RATE,
        num_bins=N_BINS,
        planning_fallacy_const=PLANNING_FALLACY_CONST,
        rewards={10: 100},
        # tasks=[
        #     Task("Task 1", time_est=1, deadline=6),
        #     Task("Task 2", time_est=2, deadline=5),
        #     Task("Task 3", time_est=3, deadline=3),
        # ],
        tasks=[
            Task("G1 - T1", time_est=2, deadline=3),
            Task("G1 - T2", time_est=4, deadline=10),
        ],
        # tasks=[
        #     Task("G1 - T1", time_est=10),
        #     Task("G1 - T2", time_est=20),
        #     # Task("Task 3", time_est=3, deadline=3),
        #     # Task("Task 4", time_est=4, deadline=8),
        # ],
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="Goal 2",
        loss_rate=LOSS_RATE,
        num_bins=N_BINS,
        planning_fallacy_const=PLANNING_FALLACY_CONST,
        rewards={10: 100},
        tasks=[
            Task("G2 - T1", time_est=1, deadline=1),
            Task("G2 - T2", time_est=3, deadline=6),
        ],
        # tasks=[
        #     Task("G2 - T1", time_est=1, deadline=1),
        #     Task("G2 - T2", time_est=2, deadline=2),
        #     # Task("Task 3", time_est=3, deadline=3),
        # ],
        unit_penalty=UNIT_PENALTY,
    )
]


example_1 = [
    Goal(
        description="Presentation",
        loss_rate=-1,
        num_bins=1,
        planning_fallacy_const=1,
        rewards={15: 100},
        tasks=[
            Task("Create PowerPoint", time_est=10, deadline=10),
            Task("Memorize lines", time_est=5, deadline=15),
        ],
        unit_penalty=1,
    )
]

example_2 = [
    Goal(
        description="Mathematics",
        loss_rate=-1,
        num_bins=1,
        planning_fallacy_const=1,
        rewards={20: 100},
        tasks=[
            Task("Solve assignment", time_est=5, deadline=10),
            Task("Prepare exam", time_est=15, deadline=20),
        ],
        unit_penalty=1,
    )
]

example_3 = [
    Goal(
        description="Misc",
        loss_rate=-1,
        num_bins=1,
        planning_fallacy_const=1,
        rewards={3: 30},
        tasks=[
            Task("Add new goals", time_est=5, deadline=2),
            Task("Write tasks in to-do list", time_est=10, deadline=1),
            # Task("Task 3", time_est=3, deadline=3),
        ],
        unit_penalty=1,
    )
]

merged_example = [
    Goal(
        description="Presentation",
        loss_rate=-1,
        num_bins=1,
        planning_fallacy_const=1,
        rewards={15: 100},
        tasks=[
            Task("Create PowerPoint", time_est=10, deadline=10),
            Task("Memorize lines", time_est=5, deadline=15),
        ],
        unit_penalty=1,
    ),
    Goal(
        description="Mathematics",
        loss_rate=-1,
        num_bins=1,
        planning_fallacy_const=1,
        rewards={20: 100},
        tasks=[
            Task("Solve assignment", time_est=5, deadline=10),
            Task("Prepare exam", time_est=15, deadline=20),
        ],
        unit_penalty=1,
    ),
    Goal(
        description="Misc",
        loss_rate=-1,
        num_bins=1,
        planning_fallacy_const=1,
        rewards={3: 30},
        tasks=[
            Task("Add new goals", time_est=5, deadline=2),
            Task("Write tasks in to-do list", time_est=10, deadline=1),
            # Task("Task 3", time_est=3, deadline=3),
        ],
        unit_penalty=1,
    )
]


def print_stats(goal):  # TODO: Move this a Goal method!
    effective_computations = goal.total_computations - goal.already_computed_pruning - goal.small_reward_pruning
    
    print(
        f"\n"
        # f"Total number of tasks: {len(goal.tasks)}\n"
        f"Total computations: {goal.total_computations}\n"
        f"Already computed: {goal.already_computed_pruning} ({goal.already_computed_pruning / goal.total_computations * 100:.3f}%)\n"
        f"Small reward: {goal.small_reward_pruning} | ({goal.small_reward_pruning / goal.total_computations * 100:.3f}%)\n"
        f"Effective computations: {effective_computations} | ({effective_computations / goal.total_computations * 100:.3f}%)\n"
    )


def local_speed_test(test_mode, n_bins: list, n_goals: list, n_tasks: list,
                     n_trials=1):
    
    OUTPUT_PATH = f"tests/smdp/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    time_results = dict()
    
    date = datetime.now()
    timestamp = f"{date.year:04d}_{date.month:02d}_{date.day:02d}_" \
                f"{date.hour:02d}_{date.minute:02d}_{date.second:02d}"
    
    time_df = pd.DataFrame()
    
    for nb in n_bins:
        for ng in n_goals:
    
            time_log = pd.Series(index=n_tasks)
    
            for nt in n_tasks:
                print(f"===== {nb} bins x {ng} goals x {nt} tasks =====")

                time_results[nt] = list()
    
                for _ in tqdm(range(n_trials)):
                    
                    goals = generate_test_case(test_mode,
                                               n_bins=nb, n_goals=ng,
                                               n_tasks=nt)
                    
                    tic = time.time()
                    to_do_list = ToDoList(goals,
                                          gamma=GAMMA,
                                          slack_reward=SLACK_REWARD)
                    to_do_list.solve(verbose=False)
                    toc = time.time()
                    
                    time_results[nt].append(toc - tic)

                # Compute mean time
                time_log[nt] = np.mean(time_results[nt])

                # Print results
                print(f"\nNumber of bins: {nb}\n"
                      f"Number of goals: {ng}\n"
                      f"Number of tasks per goal: {nt}\n"
                      f"Total number of tasks: {ng * nt}\n"
                      f"Average time: {time_log[nt]:.4f}\n"
                      f"Time results: {time_results[nt]}\n")

            time_df[ng] = time_log
            time_df.to_csv(f"{OUTPUT_PATH}/"
                           f"{timestamp}_{test_mode}_direct_time.csv")

    return time_df


def run(goals, gamma=GAMMA, verbose=False):
    
    tic = time.time()
    to_do_list = ToDoList(goals, gamma=gamma, slack_reward=SLACK_REWARD)
    to_do_list.solve(start_time=0, verbose=verbose)
    toc = time.time()

    print_stats(to_do_list)
    print()
    
    print(f"Recursive procedure took {toc - tic:.2f} seconds!")
    print()
    
    # if verbose:
    if True:
        # to_do_list.compute_pseudo_rewards(loc=0, scale=1.)
        compute_pseudo_rewards(to_do_list)
        
        optimal_policy = run_optimal_policy(to_do_list, choice_mode="max")
        
        # pprint(to_do_list.P)
        # print()
        
        # pprint(to_do_list.R)

        print(f"===== Goal level =====")
        # pprint(optimal_policy)
        # print()
        print_item(to_do_list)
        print()
        
        for goal in to_do_list.goals:
            # pprint(goal.P)
            # print()
            
            # pprint(goal.Q)
            # print()
            
            # pprint(goal.R)
            # print()
            
            # goal.compute_pseudo_rewards(start_time=0, loc=0, scale=1.)
            compute_pseudo_rewards(goal)
            goal_optimal_policy = run_optimal_policy(goal, choice_mode="max")
            
            print(f"===== {goal.description} =====")
            # pprint(goal_optimal_policy)
            # print()
            print_item(goal)
            print()
            
        # pprint(to_do_list.Q)
        # print()
        # pprint(to_do_list.F)
        # print()
        # pprint(to_do_list.R)
        # print()
        # pprint(to_do_list.R_)
        # print()

        # pprint(to_do_list.get_q_values())
        # print()
        
        # pprint(to_do_list.get_pseudo_rewards())
        # print()

    # for a, t_end in opt_P:
    #     goal = to_do_list.goals[a]
    #     goal.compute_pseudo_rewards(loc=0, scale=2.)
    #
    #     if verbose:
    #         print(goal.get_description())
    #         pprint(goal.get_q_values())
    #         pprint(goal.get_pseudo_rewards())


if __name__ == '__main__':
    # goals = [generate_goal(n_tasks=3, deadline_time=np.PINF) for s in range(1)]
    # goals = [generate_goal(n_tasks=25, deadline_time=np.PINF,
    #                        time_scale=2*s+1) for s in range(3)]
    # goals = d_bm
    # goals = [d_bm[2]]
    # goals = [d_bm[2], d_bm[3]]
    # goals = test_1
    # goals = test_2
    # goals = [single_goal]
    # goals = two_goals
    
    # run(goals=goals, verbose=False)
    
    local_speed_test(
        n_bins=[
            1,
            # 2
        ],
        n_goals=[
            # 1,
            # 2,
            # 3,
            4,
            # 5,
            # 6,
            # 7,
            # 8,
            # 9,
            # 10
        ],
        n_tasks=[
            1,
            2,
            3,
            4,
            # 25,
            # 50,
            # 75,
            # 100,
            # 125, 150, 250, 500, 750, 1000,
            # 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000,
            # 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000
        ],
        n_trials=5,
        # test_mode="averageSpeedTestSMDP",
        # test_mode="bestSpeedTestSMDP",
        test_mode="worstSpeedTestSMDP",
    )
