from collections import deque

from todolistMDP.to_do_list import Item


def generate_test_case(api_method, n_bins, n_goals, n_tasks, deadline_time=1e6,
                       penalty_rate=0, time_est=1, time_scale=1):
    
    reward = n_tasks * (n_tasks + 1)
    
    # Initialize list of goals
    goals = deque()
    
    for goal_idx in range(n_goals):
        
        # Initialize list of tasks
        tasks = deque()
        
        for task_idx in range(1, n_tasks+1):
            
            if api_method == "averageSpeedTestSMDP":
    
                time_est = task_idx * time_scale
    
                if task_idx % 10 == 0:
                    deadline = n_tasks - task_idx + 1
                else:
                    deadline = task_idx
                    
            elif api_method == "bestSpeedTestSMDP":
                deadline = task_idx
                time_est = task_idx * time_scale

            elif api_method == "exhaustiveSpeedTestSMDP":
                deadline = task_idx
                time_est = 1
                
            elif api_method == "realSpeedTestSMDP":
                deadline = task_idx
                time_est = time_est
                
            elif api_method == "worstSpeedTestSMDP":
                deadline = n_tasks - task_idx + 1
                time_est = task_idx * time_scale

            else:
                raise NotImplementedError(
                    f"Method {api_method} not implemented!")
            
            task = Item(
                f"G{goal_idx} T{task_idx}",
                deadline=deadline,
                time_est=time_est
            )
            
            tasks.append(task)

        goal = Item(
            description=f"G{goal_idx}",
            deadline=(goal_idx + 1) * deadline_time + 1,
            loss_rate=-1,
            num_bins=n_bins,
            penalty_rate=penalty_rate,
            value=reward
        )

        goal.add_items(tasks)
        
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
