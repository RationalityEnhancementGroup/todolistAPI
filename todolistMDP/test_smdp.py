import numpy as np
import sys
import time

from math import factorial
from pprint import pprint
from tqdm import tqdm

from todolistMDP.to_do_list import Task, Goal, ToDoList

# ===== Constants =====
LOSS_RATE = 0  # Default
# LOSS_RATE = 1
# LOSS_RATE = -1e-3

# GAMMA = 1 - 1e-3  # 0.9999
GAMMA = 0.9999

N_GOALS = 1
N_TASKS = 500

START_TIME = 0  # Default

TIME_SCALE = 1  # Default
# TIME_SCALE = 30

VALUE_SCALE = 1  # Default
# VALUE_SCALE = 10

SLACK_REWARD = 0  # 1e-3
TIME_PRECISION = 1
TIME_SUPPORT = None

HARD_DEADLINE = True
TASK_UNIT_PENALTY = 0.
UNIT_PENALTY = 0.

sys.setrecursionlimit(10000)

d_bm = [
    Goal(
        description="Goal A",
        # deadline=10,
        hard_deadline=HARD_DEADLINE,
        loss_rate=LOSS_RATE,
        # penalty=-10,
        # reward=100,
        rewards={TIME_SCALE * 10: VALUE_SCALE * 100},
        tasks=[
             Task("Task A1", time_est=TIME_SCALE * 1),
             Task("Task A2", time_est=TIME_SCALE * 1)
        ],
        task_unit_penalty=TASK_UNIT_PENALTY,
        time_precision=TIME_PRECISION,
        time_support=TIME_SUPPORT,
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="Goal B",
        # deadline=10,
        hard_deadline=HARD_DEADLINE,
        loss_rate=LOSS_RATE,
        # penalty=0,
        # reward=10,
        # rewards={TIME_SCALE * 10: 10},  # Simplified
        rewards={TIME_SCALE * 1: VALUE_SCALE * 10,
                 TIME_SCALE * 10: VALUE_SCALE * 10},
        tasks=[
             Task("Task B1", time_est=TIME_SCALE * 2),
             Task("Task B2", time_est=TIME_SCALE * 2)
        ],
        task_unit_penalty=TASK_UNIT_PENALTY,
        time_precision=TIME_PRECISION,
        time_support=TIME_SUPPORT,
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(description="Goal C",
         # deadline=6,
         hard_deadline=HARD_DEADLINE,
         loss_rate=LOSS_RATE,
         # penalty=-1,
         # reward=100,
         # rewards={TIME_SCALE * 6: 100},  # Simplified
         rewards={TIME_SCALE * 1: VALUE_SCALE * 10,
                  TIME_SCALE * 6: VALUE_SCALE * 100},
         tasks=[
             Task("Task C1", time_est=TIME_SCALE * 3),
             Task("Task C2", time_est=TIME_SCALE * 3)
         ],
         task_unit_penalty=TASK_UNIT_PENALTY,
         time_precision=TIME_PRECISION,
         time_support=TIME_SUPPORT,
         unit_penalty=UNIT_PENALTY,
    ),
    Goal(description="Goal D",
         # deadline=40,
         hard_deadline=HARD_DEADLINE,
         loss_rate=LOSS_RATE,
         # penalty=-10,
         # reward=10,
         # rewards={TIME_SCALE * 40: 10},  # Simplified
         rewards={TIME_SCALE * 20: VALUE_SCALE * 100,
                  TIME_SCALE * 40: VALUE_SCALE * 10},
         tasks=[
             Task("Task D1", time_est=TIME_SCALE * 3),
             Task("Task D2", time_est=TIME_SCALE * 3)
         ],
         task_unit_penalty=TASK_UNIT_PENALTY,
         time_precision=TIME_PRECISION,
         time_support=TIME_SUPPORT,
         unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="Goal E",
        # deadline=70,
        hard_deadline=HARD_DEADLINE,
        loss_rate=LOSS_RATE,
        # penalty=-110,
        # reward=10,
        # rewards={70: 10},  # Simplified
        rewards={TIME_SCALE * 60: VALUE_SCALE * 100,
                 TIME_SCALE * 70: VALUE_SCALE * 10},
        tasks=[
            Task("Task E1", time_est=TIME_SCALE * 3),
            Task("Task E2", time_est=TIME_SCALE * 3)
        ],
        task_unit_penalty=TASK_UNIT_PENALTY,
        time_precision=TIME_PRECISION,
        time_support=TIME_SUPPORT,
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="Goal F",
        # deadline=70,
        hard_deadline=HARD_DEADLINE,
        loss_rate=LOSS_RATE,
        # penalty=-110,
        # reward=10,
        # rewards={70: 10},  # Simplified
        rewards={TIME_SCALE * 60: VALUE_SCALE * 100,
                 TIME_SCALE * 70: VALUE_SCALE * 10},
        tasks=[
            Task("Task F1", time_est=TIME_SCALE * 3),
            Task("Task F2", time_est=TIME_SCALE * 3)
        ],
        task_unit_penalty=TASK_UNIT_PENALTY,
        time_precision=TIME_PRECISION,
        time_support=TIME_SUPPORT,
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
        hard_deadline=HARD_DEADLINE,
        loss_rate=LOSS_RATE,
        rewards={2000: 1000},
        tasks=[
            Task("Task 1", time_est=100),
            Task("Task 2", time_est=200),
            Task("Task 3", time_est=300),
            Task("Task 4", time_est=400),
            Task("Task 5", time_est=500)
        ],
        task_unit_penalty=TASK_UNIT_PENALTY,
        time_precision=TIME_PRECISION,
        time_support=TIME_SUPPORT,
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="G2",
        hard_deadline=HARD_DEADLINE,
        loss_rate=LOSS_RATE,
        rewards={3000: 3000},
        tasks=[
            Task("Task 1", time_est=100),
            Task("Task 2", time_est=200),
            Task("Task 3", time_est=300),
            Task("Task 4", time_est=400),
            Task("Task 5", time_est=500)
        ],
        task_unit_penalty=TASK_UNIT_PENALTY,
        time_precision=TIME_PRECISION,
        time_support=TIME_SUPPORT,
        unit_penalty=UNIT_PENALTY,
    ),
    Goal(
        description="G3",
        hard_deadline=HARD_DEADLINE,
        loss_rate=LOSS_RATE,
        rewards={4000: 5000},
        tasks=[
            Task("Task 1", time_est=100),
            Task("Task 2", time_est=200),
            Task("Task 3", time_est=300),
            Task("Task 4", time_est=400),
            Task("Task 5", time_est=500)
        ],
        task_unit_penalty=TASK_UNIT_PENALTY,
        time_precision=TIME_PRECISION,
        time_support=TIME_SUPPORT,
        unit_penalty=UNIT_PENALTY,
    )
]


def generate_discrepancy_test(n_goals, n_tasks):
    deadline = n_goals * n_tasks ** 2
    return [
        Goal(
            description=f"G{g+1}",
            hard_deadline=True,
            loss_rate=0,
            rewards={deadline: (g + 1) * 100},
            tasks=[
                Task(f"T{t+1}", time_est=t+1)
                # Task(f"T{t+1}", time_est=1)
                for t in range(n_tasks)
            ],
            task_unit_penalty=0,
            time_precision=1,
            time_support=None,
            unit_penalty=0,
        )
        for g in range(n_goals)
    ]


def generate_goal(n_tasks, deadline_time, reward=100, time_scale=TIME_SCALE):
    return Goal(
        description="__GOAL__",
        loss_rate=0,
        rewards={deadline_time: reward},
        # tasks=[
        #     Task(f"T{i}", deadline=n_tasks-i+1, time_est=i * time_scale)
        #     for i in range(1, n_tasks+1)
        # ]
        tasks=[
            Task(
                description=f"T{i}",
                deadline=n_tasks * 25,
                time_est=25
            )
            for i in range(1, n_tasks+1)
        ],
        time_precision=TIME_PRECISION,
        time_support=TIME_SUPPORT
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
                goals = [
                    generate_goal(n_tasks, deadline_time=n_tasks * n_goals * 30)
                    for s in range(n_goals)
                ]
                to_do_list = ToDoList(goals, gamma=gamma, slack_reward=0)
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
    to_do_list = ToDoList(goals, gamma=gamma, slack_reward=SLACK_REWARD)
    to_do_list.solve(start_time=0, verbose=verbose)
    toc = time.time()
    print()
    print(f"Recursive procedure took {toc - tic:.2f} seconds!")
    print()
    
    # s = (0, 0, 0, 0, 0, 0)
    
    # s = (0, 0)

    print_stats(to_do_list)
    
    # print(to_do_list.get_highest_negative_reward())
    
    # pprint(to_do_list.get_q_values(s=s))
    # pprint(to_do_list.get_q_values())
    # to_do_list.compute_pseudo_rewards(loc=0, scale=0.1)
    # pprint(to_do_list.get_pseudo_rewards())
    
    st, _ = to_do_list.run_optimal_policy(verbose=True)
    
    print()
    
    # for goal in to_do_list.get_goals():
    #     print(goal.get_description())
    #     print(goal.get_highest_negative_reward())
    #     pprint(goal.get_q_values())
    #     goal.compute_pseudo_rewards(loc=10, scale=2.)
    #     pprint(goal.get_pseudo_rewards())

    # for a, t_end in st:
    #     goal = to_do_list.goals[a]
    #     pprint(goal.Q)

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


# goals = [generate_goal(n_tasks=3, deadline_time=np.PINF) for s in range(1)]
# goals = [generate_goal(n_tasks=2, time_scale=2*s+1) for s in range(1)]
# goals = d_bm
# goals = [d_bm[2]]
# goals = test_1

goals = generate_discrepancy_test(n_goals=3, n_tasks=3)

run(goals=goals, verbose=False)

# multi_test(
#     num_goals=[2, 3, 4, 5, 6, 7, 8],
#     num_tasks=[75],
#     num_samples=3,
#     verbose=False
# )
