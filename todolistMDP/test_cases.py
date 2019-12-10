from todolistMDP.to_do_list import Goal, Task

# ===== Deterministic benchmark =====
d_bm = [
    Goal("Goal A", [
        Task("Task A1", 1),
        Task("Task A2", 1)],
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 2),
        Task("Task B2", 2)],
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal C", [
        Task("Task C1", 3),
        Task("Task C2", 3)],
        {1: 10, 6: 100},
        penalty=-1),
    Goal("Goal D", [
        Task("Task D1", 3),
        Task("Task D2", 3)],
        {20: 100, 40: 10},
        penalty=-10),
    Goal("Goal E", [
        Task("Task E1", 3),
        Task("Task E2", 3)],
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal F", [
        Task("Task F1", 3),
        Task("Task F2", 3)],
        {60: 100, 70: 10},
        penalty=-110),
]

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
p_bm = [
    Goal("CS HW", [
        Task("CS 1", time_est=1, prob=0.9),
        Task("CS 2", time_est=2, prob=0.8)],
        {5: 10},
        penalty=-10),
    Goal("EE Project", [
        Task("EE 1", time_est=4, prob=0.95),
        Task("EE 2", time_est=2, prob=0.9)],
        {10: 100},
        penalty=-200)
]

# ===== Other deterministic =====
d_1 = [
    Goal("Goal A", [
        Task("Task A1", 1)],
        {1: 100},
        penalty=-1000),
    Goal("Goal B", [
        Task("Task B2", 1)],
        {1: 10},
        penalty=-1000000),
]

d_2 = [
    Goal("Goal A", [
        Task("Task A1", 1),
        Task("Task A2", 1)],
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 1),
        Task("Task B2", 1)],
        {1: 10, 10: 10},
        penalty=0)
]

d_3 = [
    Goal("Goal A", [
        Task("Task A1", 1),
        Task("Task A2", 1)],
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 2),
        Task("Task B2", 2)],
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal C", [
        Task("Task C1", 3),
        Task("Task C2", 3)],
        {1: 10, 6: 100},
        penalty=-1),
    Goal("Goal D", [
        Task("Task D1", 3),
        Task("Task D2", 3)],
        {20: 100, 40: 10},
        penalty=-10),
]

d_4 = [
    Goal("Goal A", [
        Task("Task A1", 1),
        Task("Task A2", 1)],
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 2),
        Task("Task B2", 2)],
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal C", [
        Task("Task C1", 3),
        Task("Task C2", 3)],
        {1: 10, 6: 100},
        penalty=-1),
    Goal("Goal D", [
        Task("Task D1", 3),
        Task("Task D2", 3)],
        {20: 100, 40: 10},
        penalty=-10),
    Goal("Goal E", [
        Task("Task E1", 3),
        Task("Task E2", 3)],
        {60: 100, 70: 10},
        penalty=-110),
]

d_5 = [
    Goal("Goal A", [
        Task("Task A1", 1),
        Task("Task A2", 1)],
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 2),
        Task("Task B2", 2)],
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal C", [
        Task("Task C1", 3),
        Task("Task C2", 3)],
        {1: 10, 6: 100},
        penalty=-1),
    Goal("Goal D", [
        Task("Task D1", 3),
        Task("Task D2", 3)],
        {20: 100, 40: 10},
        penalty=-10),
    Goal("Goal E", [
        Task("Task E1", 3),
        Task("Task E2", 3)],
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal F", [
        Task("Task F1", 3),
        Task("Task F2", 3)],
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal G", [
        Task("Task G1", 3),
        Task("Task G2", 3)],
        {60: 100, 70: 10},
        penalty=-110),
]

d_6 = [
    Goal("Goal B", [
        Task("Task B1", 2),
        Task("Task B2", 2)],
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal A", [
        Task("Task A1", 1),
        Task("Task A2", 1)],
        {10: 100},
        penalty=-10),
    Goal("Goal C", [
        Task("Task C1", 3),
        Task("Task C2", 3)],
        {1: 10, 6: 100},
        penalty=-1),
    Goal("Goal D", [
        Task("Task D1", 3),
        Task("Task D2", 3)],
        {20: 100, 40: 10},
        penalty=-10),
    Goal("Goal E", [
        Task("Task E1", 3),
        Task("Task E2", 3)],
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal F", [
        Task("Task F1", 3),
        Task("Task F2", 3)],
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal G", [
        Task("Task G1", 3),
        Task("Task G2", 3)],
        {60: 100, 70: 10},
        penalty=-110)
]

d_7 = [
    Goal("CS HW", [
        Task("CS 1", time_est=1, prob=1),
        Task("CS 2", time_est=1, prob=1)],
        {3: 5},
        penalty=-10),
    Goal("EE Project", [
        Task("EE 1", time_est=1, prob=1),
        Task("EE 2", time_est=2, prob=1)],
        {4: 100},
        penalty=-200)
]

# ===== New deterministic tests (not included in the original code) =====

# Tests related to points
d_negative_reward_attainable_goals = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {10: -1000}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {20: -100}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {30: -10}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {40: -1})
]

d_unattainable_high_reward_goal = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {5: 1000}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {10: 100}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {20: 10}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {30: 1})
]

d_unattainable_low_reward_goal = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {10: 1}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {15: 1000}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {30: 10}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {30: 100})
]

# Tests related to deadlines
d_one_mixing = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {10: 1}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {25: 1}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {30: 1}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {40: 1})
]

d_negative_value_deadlines = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {-1: 1}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {-1: 1}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {-1: 1}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {-1: 1})
]

d_partially_negative_value_deadlines = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {-1: 1}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {-1: 1}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {40: 1}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {40: 1})
]

d_same_value_extra_time_deadlines = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {50: 1}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {50: 1}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {50: 1}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {50: 1})
]

d_same_value_sharp_deadlines = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {40: 1}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {40: 1}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {40: 1}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {40: 1})
]

d_same_value_unattainable_deadlines = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {1: 1}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {1: 1}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {1: 1}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {1: 1})
]

d_sharp_deadlines = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {10: 1}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {20: 1}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {30: 1}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {40: 1})
]

d_zero_value_deadlines = [
    Goal("G1",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {0: 1}),
    Goal("G2",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {0: 1}),
    Goal("G3",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {0: 1}),
    Goal("G4",
         [Task("T1", time_est=1, prob=1),
          Task("T2", time_est=2, prob=1),
          Task("T3", time_est=3, prob=1),
          Task("T4", time_est=4, prob=1)],
         {0: 1})
]

# ===== Other probabilistic =====
p_1 = [
    Goal("CS HW", [
        Task("CS 1", time_est=1, prob=0.9),
        Task("CS 2", time_est=1, prob=0.8)],
        {4: 5},
        penalty=-10),
    Goal("EE Project", [
        Task("EE 1", time_est=1, prob=0.95),
        Task("EE 2", time_est=2, prob=0.95)],
        {5: 100},
        penalty=-200)
]

p_2 = [
    Goal("CS HW", [
        Task("CS 1", time_est=1, prob=0.9),
        Task("CS 2", time_est=1, prob=0.8)],
        {3: 100},
        penalty=-200)
]

# ===== Mixed (deterministic + probabilistic) =====
m_1 = [
    Goal("CS HW", [
        Task("CS 1", time_est=1, prob=0.9),
        Task("CS 2", time_est=2, prob=0.8)],
        {7: 5},
        penalty=-10),
    Goal("EE Project", [
        Task("EE 1", time_est=4, prob=0.95),
        Task("EE 2", time_est=2, prob=0.9)],
        {14: 100},
        penalty=-200),
    Goal("Goal C", [
        Task("Task C1", 3),
        Task("Task C2", 3)],
        {1: 10, 6: 100},
        penalty=-1)
]
