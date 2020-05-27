import numpy as np
import sys
import time

from math import factorial
from pprint import pprint
from tqdm import tqdm

from todolistMDP.to_do_list import Task, Goal, ToDoList

# ===== Constants =====
LOSS_RATE = -1
GAMMA = 0.9999  # 0.9999
N_GOALS = 1
N_TASKS = 500
START_TIME = 0
TIME_SCALE = 1

sys.setrecursionlimit(10000)

d_bm = [
    Goal(
        description="Goal A",
        # deadline=10,
        loss_rate=LOSS_RATE,
        penalty=-10,
        # reward=100,
        rewards={10: 100},
        tasks=[
             Task("Task A1", time_est=1),
             Task("Task A2", time_est=1)
        ],
    ),
    Goal(
        description="Goal B",
        # deadline=10,
        loss_rate=LOSS_RATE,
        penalty=0,
        # reward=10,
        # rewards={10: 10},  # Simplified
        rewards={1: 10, 10: 10},
        tasks=[
             Task("Task B1", time_est=2),
             Task("Task B2", time_est=2)
        ]
    ),
    Goal(description="Goal C",
         # deadline=6,
         loss_rate=LOSS_RATE,
         penalty=-1,
         # reward=100,
         # rewards={6: 100},  # Simplified
         rewards={1: 10, 6: 100},
         tasks=[
             Task("Task C1", time_est=3),
             Task("Task C2", time_est=3)
         ]
    ),
    Goal(description="Goal D",
         # deadline=40,
         loss_rate=LOSS_RATE,
         penalty=-10,
         # reward=10,
         # rewards={40: 10},  # Simplified
         rewards={20: 100, 40: 10},
         tasks=[
             Task("Task D1", time_est=3),
             Task("Task D2", time_est=3)
         ]
    ),
    Goal(
        description="Goal E",
        # deadline=70,
        loss_rate=LOSS_RATE,
        penalty=-110,
        # reward=10,
        # rewards={70: 10},  # Simplified
        rewards={60: 100, 70: 10},
        tasks=[
            Task("Task E1", time_est=3),
            Task("Task E2", time_est=3)
        ],
    ),
    Goal(
        description="Goal F",
        # deadline=70,
        loss_rate=LOSS_RATE,
        penalty=-110,
        # reward=10,
        # rewards={70: 10},  # Simplified
        rewards={60: 100, 70: 10},
        tasks=[
            Task("Task F1", time_est=3),
            Task("Task F2", time_est=3)
        ],
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


def generate_goal(n_tasks, deadline_time, reward=100, time_scale=TIME_SCALE):
    return Goal(
        description="Goal",
        loss_rate=0,
        penalty=0,
        rewards={deadline_time: reward},
        tasks=[
            Task(f"T{i}", deadline=n_tasks-i+1, time_est=i * time_scale)
            for i in range(1, n_tasks+1)
        ]
    )


# def get_policy(to_do_list: ToDoList, t=0, verbose=False):
#     Q = to_do_list.get_q_values()
#
#     s = tuple(0 for _ in range(to_do_list.get_vector_length()))
#
#     st = []
#
#     r = None
#
#     while True:
#         q = Q[(s, t)]
#         q_, r_ = ToDoList.max_from_dict(q)
#
#         if r is not None:
#             print(r_ - r)
#         print(t, q_, r_, end=" | ")
#
#         if q_ is None:
#             break
#
#         st.append(q_)
#
#         a, t_ = q_
#
#         if a >= 0:
#             s = ToDoList.exec_action(s, a)
#         t = t_
#
#         r = r_
#
#     print()
#
#     return st, t


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


def multi_test(num_goals: list, num_tasks: list, num_samples=1,
               gamma=GAMMA, eval_dfs=False, verbose=False):
    
    for n_goals in num_goals:
        for n_tasks in num_tasks:
            print(f"===== {n_goals} goals x {n_tasks} tasks =====")

            time_dfs = []
            time_rec = []
            
            for _ in tqdm(range(num_samples)):
                tic = time.time()
                goals = [generate_goal(n_tasks, deadline_time=1000000 * n_goals,
                                       time_scale=2 * s + 1)
                         for s in range(n_goals)]
                to_do_list = ToDoList(goals, gamma=gamma)
                to_do_list.solve(verbose=False)
                toc = time.time()
                
                if verbose:
                    print_stats(to_do_list)
                    print(f"Recursive procedure took {toc - tic:.2f} seconds!\n")
            
                time_rec.append(toc - tic)
        
                # if eval_dfs:
                #     goal = generate_goal()
                #     tic = time.time()
                #     result = goal.dfs_solver(gamma=GAMMA, start_time=START_TIME)
                #     toc = time.time()
                #     print_stats(goal)
                #     print(f"DFS procedure took {toc - tic:.2f} seconds!")
                #
                #     time_dfs.append(toc - tic)
                #
                #     print()
                #     pprint(result["Q"])
                #     print()
    
            print(f"\nRec mean time: {np.array(time_rec).mean()}")
            print(f"Rec times {np.array(time_rec)}\n")
            
            # if eval_dfs:
            #     print(f"DFS mean time: {np.array(time_dfs).mean()}")
            #     print(f"DFS times {np.array(time_dfs}}")


def run(goals, gamma=GAMMA, verbose=False):
    
    tic = time.time()
    to_do_list = ToDoList(goals, gamma=gamma)
    to_do_list.solve(verbose=False)
    toc = time.time()
    print()
    if verbose:
        print(f"Recursive procedure took {toc - tic:.2f} seconds!")
        print()
    
    print_stats(to_do_list)
    pprint(to_do_list.get_q_values())
    st, _ = to_do_list.run_optimal_policy(verbose=True)
    print()
    
    for a, t_end in st:
        goal = to_do_list.goals[a]
        # pprint(goal.Q)

    # goal_order = [s for s, t in st]
    # print(goal_order)

    # pprint(to_do_list.P[((0, 0, 0, 0, 0, 0), 0)])

    # t = 0  # Starting time

    # for goal_idx in goal_order:
    #     goal = to_do_list.goals[goal_idx]
    #     print(f"{goal.get_description()}, Time: {t}")
    #
    #     pprint(goal.get_q_values())
    #     print()
    #
    #     # st, t = get_policy(goal, t=t, verbose=True)
    #     # print()
    #
    #     # t += goal.get_time_est()


# goals = [generate_goal(n_tasks=3, deadline_time=100) for s in range(1)]
# goals = [generate_goal(n_tasks=2, time_scale=2*s+1) for s in range(1)]
goals = d_bm
# goals = [d_bm[2]]
run(goals=goals, verbose=True)

# multi_test(
#     num_goals=[1, 2, 3, 4, 5],
#     num_tasks=[25, 50, 75, 100],
#     num_samples=1,
#     verbose=False
# )
