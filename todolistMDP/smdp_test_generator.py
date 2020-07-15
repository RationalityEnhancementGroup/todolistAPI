from collections import deque

from todolistMDP.to_do_list import Goal, Task


def generate_test_case(api_method, n_bins, n_goals, n_tasks, deadline_time=1e6,
                       time_scale=1, unit_penalty=0):
    
    reward = n_tasks * (n_tasks + 1)
    
    # Initialize list of goals
    goals = deque()
    
    for goal_idx in range(n_goals):
        
        # Initialize list of tasks
        tasks = deque()
        
        for task_idx in range(1, n_tasks+1):
            
            time_est = task_idx * time_scale
            
            if api_method == "averageSpeedTestSMDP":
                
                if task_idx % 10 == 0:
                    deadline = n_tasks - task_idx + 1
                else:
                    deadline = task_idx
                    
            elif api_method == "bestSpeedTestSMDP":
                deadline = task_idx
                
            elif api_method == "exhaustiveSpeedTestSMDP":
                deadline = task_idx
                time_est = 1
                
            elif api_method == "realSpeedTestSMDP":
                deadline = task_idx
                time_est = 5
                
            elif api_method == "worstSpeedTestSMDP":
                deadline = n_tasks - task_idx + 1
                
            else:
                raise NotImplementedError(
                    f"Method {api_method} not implemented!")
            
            task = Task(
                f"G{goal_idx} T{task_idx}",
                deadline=deadline,
                time_est=time_est
            )
            
            tasks.append(task)

        goal = Goal(
            description=f"G{goal_idx}",
            loss_rate=-1,
            num_bins=n_bins,
            rewards={deadline_time: (goal_idx + 1) * reward},
            tasks=list(tasks),
            unit_penalty=unit_penalty
        )
        
        goals.append(goal)

    return goals


def simulate_task_scheduling(to_do_list):
    q1 = deque()
    q2 = deque()
    
    for goal in to_do_list:
        for task in goal.get_tasks():
            q1.append(task)
        
    while len(q1) > 0:
        q2.append(q1.popleft())
        
    return list(q1)
