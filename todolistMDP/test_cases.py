"""
Default values:
- ID = Description
- Probability = 1
"""

from todolistMDP.to_do_list import Item

HOUR_TO_MINS = 60
DAY_TO_MINS = 24 * HOUR_TO_MINS
WEEK_TO_MINS = 7 * DAY_TO_MINS
MONTH_TO_MINS = 30 * DAY_TO_MINS
YEAR_TO_MINS = 365 * DAY_TO_MINS


def deadline_to_minutes(minutes=0, hours=0, days=0, weeks=0, months=0, years=0,
                        workload_hours=24):
    def hours_to_minutes(hours):
        return hours * 60
    
    def days_to_minutes(days, workload_hours):
        return hours_to_minutes(days * workload_hours)
    
    def weeks_to_minutes(weeks, workload_hours):
        return days_to_minutes(weeks * 7, workload_hours)
    
    def months_to_minutes(months, workload_hours):
        # We take a month to have 30 days for simplicity
        return days_to_minutes(months * 30, workload_hours)
    
    def years_to_minutes(years, workload_hours):
        return days_to_minutes(years * 365, workload_hours)
    
    minutes_to_deadline = (
        minutes + hours_to_minutes(hours) +
        days_to_minutes(days, workload_hours) +
        weeks_to_minutes(weeks, workload_hours) +
        months_to_minutes(months, workload_hours) +
        years_to_minutes(years, workload_hours)
    )
    
    return minutes_to_deadline
    

# ===== Deterministic benchmark =====
# d_bm = [
#     Goal(description="Goal A",
#          tasks=[Task("Task A1", 1),
#                 Task("Task A2", 1)],
#          rewards={10: 100},
#          penalty=-10),
#     Goal(description="Goal B",
#          tasks=[Task("Task B1", 2),
#                 Task("Task B2", 2)],
#          rewards={1: 10, 10: 10},
#          penalty=0),
#     Goal(description="Goal C",
#          tasks=[Task("Task C1", 3),
#                 Task("Task C2", 3)],
#          rewards={1: 10, 6: 100},
#          penalty=-1),
#     Goal(description="Goal D",
#          tasks=[Task("Task D1", 3),
#                 Task("Task D2", 3)],
#          rewards={20: 100, 40: 10},
#          penalty=-10),
#     Goal(description="Goal E",
#          tasks=[Task("Task E1", 3),
#                 Task("Task E2", 3)],
#          rewards={60: 100, 70: 10},
#          penalty=-110),
#     Goal(description="Goal F",
#          tasks=[Task("Task F1", 3),
#                 Task("Task F2", 3)],
#          rewards={60: 100, 70: 10},
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

d_bm_mod = [
    Item(
        description="Goal A",
        deadline=10,
        # penalty=-10,
        reward=100,
        # rewards={10: 100},
        sub_items=[
            Item(
                description="Task A1",
                time_est=1
            ),
            Item(
                description="Task A2",
                time_est=1
            )
        ]
    ),
    Item(
        description="Goal B",
        deadline=10,
        # penalty=0,
        reward=10,
        # rewards={1: 10, 10: 10},
        sub_items=[
            Item(
                description="Task B1",
                time_est=2
            ),
            Item(
                description="Task B2",
                time_est=2
            )
        ],
    ),
    Item(
        description="Goal C",
        deadline=6,
        # penalty=-1,
        reward=100,
        # rewards={1: 10, 6: 100},
        sub_items=[
            Item(
                description="Task C1",
                time_est=3
            ),
            Item(
                description="Task C2",
                time_est=3
            )
        ],
    ),
    Item(
        description="Goal D",
        deadline=40,
        # penalty=-10,
        reward=10,
        # rewards={20: 100, 40: 10},
        sub_items=[
            Item(
                description="Task D1",
                time_est=3
            ),
            Item(
                description="Task D2",
                time_est=3
            )
        ],
    ),
    Item(
        description="Goal E",
        deadline=70,
        # penalty=-110,
        reward=10,
        # rewards={60: 100, 70: 10},
        sub_items=[
            Item(
                description="Task E1",
                time_est=3
            ),
            Item(
                description="Task E2",
                time_est=3
            )
        ],
    ),
    Item(
        description="Goal F",
        deadline=70,
        # penalty=-110
        reward=10,
        # rewards={60: 100, 70: 10},
        sub_items=[
            Item(
                description="Task F1",
                time_est=3
            ),
            Item(
                description="Task F2",
                time_est=3
            )
        ],
    ),
]


# # ===== Probabilistic benchmark =====
# p_bm = [
#     Goal(description="CS HW",
#          tasks=[Task("CS 1", time_est=1, prob=0.9),
#                 Task("CS 2", time_est=2, prob=0.8)],
#          rewards={5: 10},
#          penalty=-10),
#     Goal(description="EE Project",
#          tasks=[Task("EE 1", time_est=4, prob=0.95),
#                 Task("EE 2", time_est=2, prob=0.9)],
#          rewards={10: 100},
#          penalty=-200)
# ]
#
# # ===== Other deterministic =====
# d_1 = [
#     Goal(description="Goal A",
#          tasks=[Task("Task A1", 1)],
#          rewards={1: 100},
#          penalty=-1000),
#     Goal(description="Goal B",
#          tasks=[Task("Task B2", 1)],
#          rewards={1: 10},
#          penalty=-1000000),
# ]
#
# d_2 = [
#     Goal(description="Goal A",
#          tasks=[Task("Task A1", 1),
#                 Task("Task A2", 1)],
#          rewards={10: 100},
#          penalty=-10),
#     Goal(description="Goal B",
#          tasks=[Task("Task B1", 1),
#                 Task("Task B2", 1)],
#          rewards={1: 10, 10: 10},
#          penalty=0)
# ]
#
# d_3 = [
#     Goal(description="Goal A",
#          tasks=[Task("Task A1", 1),
#                 Task("Task A2", 1)],
#          rewards={10: 100},
#          penalty=-10),
#     Goal(description="Goal B",
#          tasks=[Task("Task B1", 2),
#                 Task("Task B2", 2)],
#          rewards={1: 10, 10: 10},
#          penalty=0),
#     Goal(description="Goal C",
#          tasks=[Task("Task C1", 3),
#                 Task("Task C2", 3)],
#          rewards={1: 10, 6: 100},
#          penalty=-1),
#     Goal(description="Goal D",
#          tasks=[Task("Task D1", 3),
#                 Task("Task D2", 3)],
#          rewards={20: 100, 40: 10},
#          penalty=-10),
# ]
#
# d_4 = [
#     Goal(description="Goal A",
#          tasks=[Task("Task A1", 1),
#                 Task("Task A2", 1)],
#          rewards={10: 100},
#          penalty=-10),
#     Goal(description="Goal B",
#          tasks=[Task("Task B1", 2),
#                 Task("Task B2", 2)],
#          rewards={1: 10, 10: 10},
#          penalty=0),
#     Goal(description="Goal C",
#          tasks=[Task("Task C1", 3),
#                 Task("Task C2", 3)],
#          rewards={1: 10, 6: 100},
#          penalty=-1),
#     Goal(description="Goal D",
#          tasks=[Task("Task D1", 3),
#                 Task("Task D2", 3)],
#          rewards={20: 100, 40: 10},
#          penalty=-10),
#     Goal(description="Goal E",
#          tasks=[Task("Task E1", 3),
#                 Task("Task E2", 3)],
#          rewards={60: 100, 70: 10},
#          penalty=-110),
# ]
#
# d_5 = [
#     Goal(description="Goal A",
#          tasks=[Task("Task A1", 1),
#                 Task("Task A2", 1)],
#          rewards={10: 100},
#          penalty=-10),
#     Goal(description="Goal B",
#          tasks=[Task("Task B1", 2),
#                 Task("Task B2", 2)],
#          rewards={1: 10, 10: 10},
#          penalty=0),
#     Goal(description="Goal C",
#          tasks=[Task("Task C1", 3),
#                 Task("Task C2", 3)],
#          rewards={1: 10, 6: 100},
#          penalty=-1),
#     Goal(description="Goal D",
#          tasks=[Task("Task D1", 3),
#                 Task("Task D2", 3)],
#          rewards={20: 100, 40: 10},
#          penalty=-10),
#     Goal(description="Goal E",
#          tasks=[Task("Task E1", 3),
#                 Task("Task E2", 3)],
#          rewards={60: 100, 70: 10},
#          penalty=-110),
#     Goal(description="Goal F",
#          tasks=[Task("Task F1", 3),
#                 Task("Task F2", 3)],
#          rewards={60: 100, 70: 10},
#          penalty=-110),
#     Goal(description="Goal G",
#          tasks=[Task("Task G1", 3),
#                 Task("Task G2", 3)],
#          rewards={60: 100, 70: 10},
#          penalty=-110),
# ]
#
# d_6 = [
#     Goal(description="Goal B",
#          tasks=[Task("Task B1", 2),
#                 Task("Task B2", 2)],
#          rewards={1: 10, 10: 10},
#          penalty=0),
#     Goal(description="Goal A",
#          tasks=[Task("Task A1", 1),
#                 Task("Task A2", 1)],
#          rewards={10: 100},
#          penalty=-10),
#     Goal(description="Goal C",
#          tasks=[Task("Task C1", 3),
#                 Task("Task C2", 3)],
#          rewards={1: 10, 6: 100},
#          penalty=-1),
#     Goal(description="Goal D",
#          tasks=[Task("Task D1", 3),
#                 Task("Task D2", 3)],
#          rewards={20: 100, 40: 10},
#          penalty=-10),
#     Goal(description="Goal E",
#          tasks=[Task("Task E1", 3),
#                 Task("Task E2", 3)],
#          rewards={60: 100, 70: 10},
#          penalty=-110),
#     Goal(description="Goal F",
#          tasks=[Task("Task F1", 3),
#                 Task("Task F2", 3)],
#          rewards={60: 100, 70: 10},
#          penalty=-110),
#     Goal(description="Goal G",
#          tasks=[Task("Task G1", 3),
#                 Task("Task G2", 3)],
#          rewards={60: 100, 70: 10},
#          penalty=-110)
# ]
#
# d_7 = [
#     Goal(description="CS HW",
#          tasks=[Task(description="CS 1", time_est=1),
#                 Task(description="CS 2", time_est=1)],
#          rewards={3: 5},
#          penalty=-10),
#     Goal(description="EE Project",
#          tasks=[Task(description="EE 1", time_est=1),
#                 Task(description="EE 2", time_est=2)],
#          rewards={4: 100},
#          penalty=-200)
# ]
#
#
# # ===== New deterministic tests (not included in the original code) =====
# def generate_deterministic_test(num_goals, num_tasks, goal_reward=1,
#                                 task_time_est=1, goal_deadline=None):
#     if goal_deadline is None:
#         goal_deadline = num_goals * num_tasks * task_time_est
#
#     return [
#         Goal(description=f"G{goal_idx + 1}",
#              tasks=[
#                  Task(description=f"T{task_idx + 1}", time_est=task_time_est)
#                  for task_idx in range(num_tasks)
#              ],
#              rewards={goal_deadline: goal_reward})
#         for goal_idx in range(num_goals)
#     ]
#
#
# # Tests related to points
# d_negative_reward_attainable_goals = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={10: -1000}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={20: -100}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={30: -10}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={40: -1})
# ]
#
# d_unattainable_high_reward_goal = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={5: 1000}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={10: 100}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={20: 10}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={30: 1})
# ]
#
# d_unattainable_low_reward_goal = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={10: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={15: 1000}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={30: 10}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={30: 100})
# ]
#
# # Tests related to deadlines
# d_all_different_extra_time_deadlines = [
#     Goal(description="G1",
#          tasks=[
#             Task(description="T1", time_est=1),
#             Task(description="T2", time_est=2),
#             Task(description="T3", time_est=3),
#             Task(description="T4", time_est=4)
#          ],
#          rewards={15: 1}),
#     Goal(description="G2",
#          tasks=[
#             Task(description="T1", time_est=1),
#             Task(description="T2", time_est=2),
#             Task(description="T3", time_est=3),
#             Task(description="T4", time_est=4)
#          ],
#          rewards={25: 1}),
#     Goal(description="G3",
#          tasks=[
#             Task(description="T1", time_est=1),
#             Task(description="T2", time_est=2),
#             Task(description="T3", time_est=3),
#             Task(description="T4", time_est=4)
#          ],
#          rewards={35: 1}),
#     Goal(description="G4",
#          tasks=[
#             Task(description="T1", time_est=1),
#             Task(description="T2", time_est=2),
#             Task(description="T3", time_est=3),
#             Task(description="T4", time_est=4)
#          ],
#          rewards={45: 1})
# ]
#
# d_distant_deadlines = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={1 * YEAR_TO_MINS: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={2 * YEAR_TO_MINS: 1}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={3 * YEAR_TO_MINS: 1}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={4 * YEAR_TO_MINS: 1}),
#     Goal(description="G5",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={5 * YEAR_TO_MINS: 1}),
#     Goal(description="G6",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={6 * YEAR_TO_MINS: 1}),
#     Goal(description="G7",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={7 * YEAR_TO_MINS: 1}),
#     Goal(description="G8",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={8 * YEAR_TO_MINS: 1}),
#     Goal(description="G9",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={9 * YEAR_TO_MINS: 1}),
#     Goal(description="G10",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={10 * YEAR_TO_MINS: 1})
# ]
#
# d_one_mixing = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={10: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={25: 1}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={30: 1}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={40: 1})
# ]
#
# d_negative_value_deadlines = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={-1: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={-1: 1}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={-1: 1}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={-1: 1})
# ]
#
# d_partially_negative_value_deadlines = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={-1: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={-1: 1}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={40: 1}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={40: 1})
# ]
#
# d_same_value_extra_time_deadlines = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={50: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={50: 1}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={50: 1}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={50: 1})
# ]
#
# d_same_value_sharp_deadlines = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={40: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={40: 1}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={40: 1}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={40: 1})
# ]
#
# d_same_value_unattainable_deadlines = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={1: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={1: 1}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={1: 1}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={1: 1})
# ]
#
# d_sharp_deadlines = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={10: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={20: 1}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={30: 1}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={40: 1})
# ]
#
# d_zero_value_deadlines = [
#     Goal(description="G1",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={0: 1}),
#     Goal(description="G2",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={0: 1}),
#     Goal(description="G3",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={0: 1}),
#     Goal(description="G4",
#          tasks=[Task(description="T1", time_est=1),
#                 Task(description="T2", time_est=2),
#                 Task(description="T3", time_est=3),
#                 Task(description="T4", time_est=4)],
#          rewards={0: 1})
# ]
#
# deterministic_tests = list(
#     d_negative_reward_attainable_goals
# )
#
# # ===== Other probabilistic =====
# p_1 = [
#     Goal(description="CS HW",
#          tasks=[Task("CS 1", time_est=1, prob=0.9),
#                 Task("CS 2", time_est=1, prob=0.8)],
#          rewards={4: 5},
#          penalty=-10),
#     Goal(description="EE Project",
#          tasks=[Task("EE 1", time_est=1, prob=0.95),
#                 Task("EE 2", time_est=2, prob=0.95)],
#          rewards={5: 100},
#          penalty=-200)
# ]
#
# p_2 = [
#     Goal(description="CS HW",
#          tasks=[Task("CS 1", time_est=1, prob=0.9),
#                 Task("CS 2", time_est=1, prob=0.8)],
#          rewards={3: 100},
#          penalty=-200)
# ]
#
# # ===== Mixed (deterministic + probabilistic) =====
# m_1 = [
#     Goal(description="CS HW",
#          tasks=[Task("CS 1", time_est=1, prob=0.9),
#                 Task("CS 2", time_est=2, prob=0.8)],
#          rewards={7: 5},
#          penalty=-10),
#     Goal(description="EE Project",
#          tasks=[Task("EE 1", time_est=4, prob=0.95),
#                 Task("EE 2", time_est=2, prob=0.9)],
#          rewards={14: 100},
#          penalty=-200),
#     Goal(description="Goal C",
#          tasks=[Task("Task C1", 3),
#                 Task("Task C2", 3)],
#          rewards={1: 10, 6: 100},
#          penalty=-1)
# ]
