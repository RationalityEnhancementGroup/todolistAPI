from collections import deque

from todolistMDP.to_do_list import Goal, Task


def generate_test_case(n_bins, n_goals, n_tasks, deadline_time=1000000,
                       reward=1000000, time_scale=1, worst=True):
    return [
        Goal(
            description=f"G{goal_idx}",
            loss_rate=-1,
            num_bins=n_bins,
            rewards={deadline_time: reward},
            tasks=[
                Task(
                    f"G{goal_idx} T{task_idx}",
                    deadline=n_tasks-task_idx+1 if worst else task_idx,
                    time_est=task_idx * time_scale
                )
                for task_idx in range(1, n_tasks+1)
            ]
        )
        for goal_idx in range(1, n_goals+1)
    ]


def simulate_task_scheduling(to_do_list):
    q1 = deque()
    q2 = deque()
    
    for goal in to_do_list:
        for task in goal.get_tasks():
            q1.append(task)
        
    while len(q1) > 0:
        q2.append(q1.popleft())
        
    return list(q1)
