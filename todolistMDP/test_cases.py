from todolistMDP.to_do_list import Goal, Task

HOUR_TO_MINS = 60
DAY_TO_MINS = 24 * HOUR_TO_MINS
WEEK_TO_MINS = 7 * DAY_TO_MINS
MONTH_TO_MINS = 30 * DAY_TO_MINS
YEAR_TO_MINS = 365 * DAY_TO_MINS

# ===== Deterministic benchmark =====
# d_bm = [
#     Goal("Goal A", [
#         Task("Task A1", 1),
#         Task("Task A2", 1)],
#          reward={10: 100},
#          penalty=-10),
#     Goal("Goal B", [
#         Task("Task B1", 2),
#         Task("Task B2", 2)],
#          reward={1: 10, 10: 10},
#          penalty=0),
#     Goal("Goal C", [
#         Task("Task C1", 3),
#         Task("Task C2", 3)],
#          reward={1: 10, 6: 100},
#          penalty=-1),
#     Goal("Goal D", [
#         Task("Task D1", 3),
#         Task("Task D2", 3)],
#          reward={20: 100, 40: 10},
#          penalty=-10),
#     Goal("Goal E", [
#         Task("Task E1", 3),
#         Task("Task E2", 3)],
#          reward={60: 100, 70: 10},
#          penalty=-110),
#     Goal("Goal F", [
#         Task("Task F1", 3),
#         Task("Task F2", 3)],
#          reward={60: 100, 70: 10},
#          penalty=-110),
# ]

'''
Optimal Ordering:
 1. Task C1
 2. Task C2
 3. Task A1
 4. Task A2
 5. Task B1
 6. Task B2
 7. Task D1
 8. Task D2
 9. Task E1
10. Task E2
11. Task F1
12. Task F2
'''

# ===== Probabilistic benchmark =====
# p_bm = [
#     Goal("CS HW", [
#         Task("CS 1", time_est=1, prob=0.9),
#         Task("CS 2", time_est=2, prob=0.8)],
#          reward={5: 10},
#          penalty=-10),
#     Goal("EE Project", [
#         Task("EE 1", time_est=4, prob=0.95),
#         Task("EE 2", time_est=2, prob=0.9)],
#          reward={10: 100},
#          penalty=-200)
# ]

# ===== Other deterministic =====
# d_1 = [
#     Goal("Goal A", [
#         Task("Task A1", 1)],
#          reward={1: 100},
#          penalty=-1000),
#     Goal("Goal B", [
#         Task("Task B2", 1)],
#          reward={1: 10},
#          penalty=-1000000),
# ]

# d_2 = [
#     Goal("Goal A", [
#         Task("Task A1", 1),
#         Task("Task A2", 1)],
#          reward={10: 100},
#          penalty=-10),
#     Goal("Goal B", [
#         Task("Task B1", 1),
#         Task("Task B2", 1)],
#          reward={1: 10, 10: 10},
#          penalty=0)
# ]

# d_3 = [
#     Goal("Goal A", [
#         Task("Task A1", 1),
#         Task("Task A2", 1)],
#          reward={10: 100},
#          penalty=-10),
#     Goal("Goal B", [
#         Task("Task B1", 2),
#         Task("Task B2", 2)],
#          reward={1: 10, 10: 10},
#          penalty=0),
#     Goal("Goal C", [
#         Task("Task C1", 3),
#         Task("Task C2", 3)],
#          reward={1: 10, 6: 100},
#          penalty=-1),
#     Goal("Goal D", [
#         Task("Task D1", 3),
#         Task("Task D2", 3)],
#          reward={20: 100, 40: 10},
#          penalty=-10),
# ]

# d_4 = [
#     Goal("Goal A", [
#         Task("Task A1", 1),
#         Task("Task A2", 1)],
#          reward={10: 100},
#          penalty=-10),
#     Goal("Goal B", [
#         Task("Task B1", 2),
#         Task("Task B2", 2)],
#          reward={1: 10, 10: 10},
#          penalty=0),
#     Goal("Goal C", [
#         Task("Task C1", 3),
#         Task("Task C2", 3)],
#          reward={1: 10, 6: 100},
#          penalty=-1),
#     Goal("Goal D", [
#         Task("Task D1", 3),
#         Task("Task D2", 3)],
#          reward={20: 100, 40: 10},
#          penalty=-10),
#     Goal("Goal E", [
#         Task("Task E1", 3),
#         Task("Task E2", 3)],
#          reward={60: 100, 70: 10},
#          penalty=-110),
# ]

# d_5 = [
#     Goal("Goal A", [
#         Task("Task A1", 1),
#         Task("Task A2", 1)],
#          reward={10: 100},
#          penalty=-10),
#     Goal("Goal B", [
#         Task("Task B1", 2),
#         Task("Task B2", 2)],
#          reward={1: 10, 10: 10},
#          penalty=0),
#     Goal("Goal C", [
#         Task("Task C1", 3),
#         Task("Task C2", 3)],
#          reward={1: 10, 6: 100},
#          penalty=-1),
#     Goal("Goal D", [
#         Task("Task D1", 3),
#         Task("Task D2", 3)],
#          reward={20: 100, 40: 10},
#          penalty=-10),
#     Goal("Goal E", [
#         Task("Task E1", 3),
#         Task("Task E2", 3)],
#          reward={60: 100, 70: 10},
#          penalty=-110),
#     Goal("Goal F", [
#         Task("Task F1", 3),
#         Task("Task F2", 3)],
#          reward={60: 100, 70: 10},
#          penalty=-110),
#     Goal("Goal G", [
#         Task("Task G1", 3),
#         Task("Task G2", 3)],
#          reward={60: 100, 70: 10},
#          penalty=-110),
# ]

# d_6 = [
#     Goal("Goal B", [
#         Task("Task B1", 2),
#         Task("Task B2", 2)],
#          reward={1: 10, 10: 10},
#          penalty=0),
#     Goal("Goal A", [
#         Task("Task A1", 1),
#         Task("Task A2", 1)],
#          reward={10: 100},
#          penalty=-10),
#     Goal("Goal C", [
#         Task("Task C1", 3),
#         Task("Task C2", 3)],
#          reward={1: 10, 6: 100},
#          penalty=-1),
#     Goal("Goal D", [
#         Task("Task D1", 3),
#         Task("Task D2", 3)],
#          reward={20: 100, 40: 10},
#          penalty=-10),
#     Goal("Goal E", [
#         Task("Task E1", 3),
#         Task("Task E2", 3)],
#          reward={60: 100, 70: 10},
#          penalty=-110),
#     Goal("Goal F", [
#         Task("Task F1", 3),
#         Task("Task F2", 3)],
#          reward={60: 100, 70: 10},
#          penalty=-110),
#     Goal("Goal G", [
#         Task("Task G1", 3),
#         Task("Task G2", 3)],
#          reward={60: 100, 70: 10},
#          penalty=-110)
# ]

# d_7 = [
#     Goal("CS HW", [
#         Task("CS 1", time_est=1, prob=1),
#         Task("CS 2", time_est=1, prob=1)],
#          reward={3: 5},
#          penalty=-10),
#     Goal("EE Project", [
#         Task("EE 1", time_est=1, prob=1),
#         Task("EE 2", time_est=2, prob=1)],
#          reward={4: 100},
#          penalty=-200)
# ]

# ===== New deterministic tests (not included in the original code) =====

# Tests related to points
d_negative_reward_attainable_goals = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={10: -1000}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={20: -100}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={30: -10}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={40: -1})
]

d_unattainable_high_reward_goal = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={5: 1000}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={10: 100}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={20: 10}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={30: 1})
]

d_unattainable_low_reward_goal = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={10: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={15: 1000}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={30: 10}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={30: 100})
]

# Tests related to deadlines
d_different_value_extra_time_deadlines = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={15: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={25: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={35: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={45: 1})
]

d_distant_deadlines = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={1 * YEAR_TO_MINS: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={2 * YEAR_TO_MINS: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={3 * YEAR_TO_MINS: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={4 * YEAR_TO_MINS: 1}),
    Goal(description="G5", goal_id="G5",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={5 * YEAR_TO_MINS: 1}),
    Goal(description="G6", goal_id="G6",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={6 * YEAR_TO_MINS: 1}),
    Goal(description="G7", goal_id="G7",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={7 * YEAR_TO_MINS: 1}),
    Goal(description="G8", goal_id="G8",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={8 * YEAR_TO_MINS: 1}),
    Goal(description="G9", goal_id="G9",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={9 * YEAR_TO_MINS: 1}),
    Goal(description="G10", goal_id="G10",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={10 * YEAR_TO_MINS: 1})
]

d_one_mixing = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={10: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={25: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={30: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={40: 1})
]

d_negative_value_deadlines = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={-1: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={-1: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={-1: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={-1: 1})
]

d_partially_negative_value_deadlines = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={-1: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={-1: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={40: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={40: 1})
]

d_same_value_extra_time_deadlines = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={50: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={50: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={50: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={50: 1})
]

d_same_value_sharp_deadlines = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={40: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={40: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={40: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={40: 1})
]

d_same_value_unattainable_deadlines = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={1: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={1: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={1: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={1: 1})
]

d_sharp_deadlines = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={10: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={20: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={30: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={40: 1})
]

d_zero_value_deadlines = [
    Goal(description="G1", goal_id="G1",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={0: 1}),
    Goal(description="G2", goal_id="G2",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={0: 1}),
    Goal(description="G3", goal_id="G3",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={0: 1}),
    Goal(description="G4", goal_id="G4",
         tasks=[Task(description="T1", task_id="T1", time_est=1, prob=1),
                Task(description="T2", task_id="T2", time_est=2, prob=1),
                Task(description="T3", task_id="T3", time_est=3, prob=1),
                Task(description="T4", task_id="T4", time_est=4, prob=1)],
         reward={0: 1})
]

# ===== Other probabilistic =====
# p_1 = [
#     Goal("CS HW", [
#         Task("CS 1", time_est=1, prob=0.9),
#         Task("CS 2", time_est=1, prob=0.8)],
#          reward={4: 5},
#          penalty=-10),
#     Goal("EE Project", [
#         Task("EE 1", time_est=1, prob=0.95),
#         Task("EE 2", time_est=2, prob=0.95)],
#          reward={5: 100},
#          penalty=-200)
# ]

# p_2 = [
#     Goal("CS HW", [
#         Task("CS 1", time_est=1, prob=0.9),
#         Task("CS 2", time_est=1, prob=0.8)],
#          reward={3: 100},
#          penalty=-200)
# ]

# ===== Mixed (deterministic + probabilistic) =====
# m_1 = [
#     Goal("CS HW", [
#         Task("CS 1", time_est=1, prob=0.9),
#         Task("CS 2", time_est=2, prob=0.8)],
#          reward={7: 5},
#          penalty=-10),
#     Goal("EE Project", [
#         Task("EE 1", time_est=4, prob=0.95),
#         Task("EE 2", time_est=2, prob=0.9)],
#          reward={14: 100},
#          penalty=-200),
#     Goal("Goal C", [
#         Task("Task C1", 3),
#         Task("Task C2", 3)],
#          reward={1: 10, 6: 100},
#          penalty=-1)
# ]
