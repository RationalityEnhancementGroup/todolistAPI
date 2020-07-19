import numpy as np

from collections import deque
from pprint import pprint
from todolistMDP.to_do_list import ToDoList, Goal


def compute_start_state_pseudo_rewards(to_do_list, bias=None, scale=None):
    
    # Get list of goals
    goals = to_do_list.get_goals()
    
    # Get goal future Q-values
    # future_q = dict()  # {Goal index: Future value after exec in s[0]}
    task_q = dict()  # {Task ID: Q-value for task execution in s[0]}
    next_q = dict()  # {Task ID: Q-value after task execution in s[0]}
    
    # Initialize best Q-value
    best_q = np.NINF
    best_next_q = None
    
    # Initialize list of incentivized tasks
    incentivized_tasks = deque()
    
    for goal in goals:
        
        # Get slack reward
        slack_reward = goal.compute_slack_reward(0)

        if best_q <= slack_reward:
            best_q = slack_reward
            best_next_q = 0
        
        # Get time step after executing goal
        t_ = goal.get_time_est()
        
        # Set future Q-value
        # future_q[goal_idx] = goal.get_future_q(t_)
        future_q = goal.get_future_q(t_)
        
        # Initialize (s)tate and (t)ime
        s = goal.get_start_state()
        t = 0
        
        # Get all tasks
        tasks = goal.get_tasks()
        
        for a in goal.get_q_values(s, t):
            
            if a != -1:
                
                # Get task object
                task = tasks[a]
                
                # Compute task Q-value
                q = goal.get_q_values(s, t, a) + future_q
                
                # Get task ID
                task_id = task.get_id()

                # Store Q-value for task execution in s[0]
                task_q[task_id] = q
                
                # Get expected task loss
                loss = goal.R[s][t][a]
                
                # Set expected task loss
                task.set_expected_loss(loss)
    
                # Compute Q-value after transition
                next_q[task_id] = q - loss
                
                # Update best Q-value and best next Q-value
                if best_q <= q:
                    best_q = q
                    best_next_q = q - loss
    
                # Add tasks to the list of incentivized tasks (?!)
                incentivized_tasks.append(task)
                
    # Initialize minimum pseudo-reward value (bias for linear transformation)
    min_pr = 0

    # Initialize sum of pseudo-rewards
    sum_pr = 0
    
    # Compute untransformed pseudo-rewards
    for task in incentivized_tasks:
        
        # Get task ID
        task_id = task.get_id()
        
        # Get expected task loss
        task_loss = task.get_expected_loss()
        
        # Compute pseudo-reward
        pr = task_q[task_id] - best_next_q + task_loss
        
        # Store pseudo-reward
        task.set_optimal_reward(pr)
        
        # Update minimum pseudo-reward
        min_pr = min(min_pr, pr)
        
        # Update sum of pseudo-rewards
        sum_pr += pr
        
    # Compute sum of goal values
    sum_goal_values = sum([goal.get_reward() for goal in goals])

    # Update sum of pseudo-rewards
    sum_pr += len(incentivized_tasks) * (1 - min_pr)  # min_pr < 0 (!)
    
    # Define scaling and shifting parameters
    if bias is None or scale is None:
        scale = sum_goal_values / sum_pr
        bias = (1 - min_pr) * scale
        
    print(bias, scale)

    # Initialize {Task ID: pseudo-reward} dictionary
    id2pr = dict()

    # Sanity check for the sum of pseudo-rewards
    sc_sum_pr = 0

    # Perform linear transformation on task pseudo-rewards
    # for task in optimal_tasks + suboptimal_tasks + slack_tasks:
    for task in incentivized_tasks:
        
        # Get task unique identification
        task_id = task.get_id()
    
        # Transform pseudo-reward
        pr = scale * task.get_optimal_reward() + bias
    
        # Store new (zero) pseudo-reward
        task.set_optimal_reward(pr)
    
        # If task is not slack action
        if task.get_idx() != -1:
            
            # Update sanity check for the sum of pseudo-rewards
            sc_sum_pr += task.get_optimal_reward()
    
        # Store pseudo-reward {Task ID: pseudo-reward}
        id2pr[task_id] = pr
        
    # Initialize tasks queue
    optimal_tasks = deque()
    suboptimal_tasks = deque()
    slack_tasks = deque()
    
    # Run goal-level optimal policy in order to get optimal sequence of goals
    P, t = run_optimal_policy(to_do_list)
    
    for entry in P:
        
        # Get next (a)ction and initial (t)ime
        t = entry["t"]
        a = entry["a"]
        
        # If next action is not slack-off action
        if a != -1:
            
            # Get goal that correspond to that (a)ction
            goal = goals[a]
            
            # Get best action
            best_a = goal.get_best_action(t)
            
            # Get all goal's tasks
            tasks = goal.get_tasks()
            
            for task in goal.sorted_tasks_by_deadlines:
                
                # Get task ID
                task_id = task.get_id()
                
                # Get task index
                task_idx = task.get_idx()
                
                # Get task object for the corresponding index
                task = tasks[task_idx]
                
                # If the pseudo-reward has already been computed, assign it
                if task_id in id2pr.keys():
                    
                    # Append task to the queue of tasks to be scheduled
                    if best_a != -1:
                        optimal_tasks.append(task)
                    else:
                        suboptimal_tasks.append(task)
                        
                    # Set transformed task pseudo-reward as optimal value
                    task.set_optimal_reward(id2pr[task_id])
                
            # If the goal is worth pursuing (i.e. slack action is not the best)
            if best_a == -1:

                # Get slack action associated with current goal
                slack_action = goal.get_slack_action()
                
                # Set optimal reward
                slack_action.set_optimal_reward(0)
                
                # Add slack action to the list of slack tasks
                slack_tasks.append(slack_action)
        
    return {
        "optimal_tasks": optimal_tasks,
        "suboptimal_tasks": suboptimal_tasks,
        "slack_tasks": slack_tasks,
        
        "id2pr": id2pr,
        "sc_sum_pr": sc_sum_pr,
        "scale": scale,
        "bias": bias
    }


def run_optimal_policy(obj, s=None, t=0, choice_mode="random"):
    """
    Runs optimal policy.

    Args:
        obj: {Goal, ToDoList}
        s: Initial state (binary vector of task completion).
        t: Initial time (non-negative integer).
        choice_mode: {"max", "random"}

    Returns:
        (Optimal policy, Time after executing policy)
    """
    
    # Get items that belong to the given objects
    if type(obj) is Goal:
        items = obj.get_tasks()
    elif type(obj) is ToDoList:
        items = obj.get_goals()
    else:
        raise NotImplementedError(
            f"Unknown object type {type(obj)}")
    
    # If no state was provided, get start state
    if s is None:
        s = tuple(0 for _ in range(len(items)))
    
    # Check whether the state has a valid length (in case it is user-provided)
    assert len(s) == len(items)
    
    # Initialize optimal-policy actions list
    optimal_policy = deque()
    
    while True:
        
        # Get next action according to the optimal policy
        a = obj.P[s][t]
        
        # If next action is termination or slack-off action
        if a is None:
            break
        
        # Get action item object
        if a != -1:
            item = items[a]
        else:
            item = obj.get_slack_action()
        
        # Set optimal reward for the action
        # item.set_optimal_reward(obj.PR[s][t][a]["E"])
        
        # Append (action, time') to the optimal-policy actions list
        optimal_policy.append({
            "s":   s,
            "a":   a,
            "t":   t,
            # "t_": t_,
            "obj": item,
            # "PR":  obj.PR[s][t][a]["E"]
        })
        
        # Break if next action is slack-off
        if a == -1:
            break
        
        # Get time transitions
        time_transitions = item.get_time_transitions()
        times = sorted(list(time_transitions.keys()))
        values = [time_transitions[t] for t in times]
        
        # Get item time estimate in a most-likely or random manner
        if choice_mode == "max":
            idx = int(np.argmax(values))
            time_est = times[idx]
        
        elif choice_mode == "random":
            time_est = \
                np.random.choice(times, size=1, replace=False, p=values)[0]
        
        else:
            raise NotImplementedError(f"Unsupported choice mode {choice_mode}!")
        
        # Compute next time transition
        t_ = t + time_est
        
        # Move to the next time step
        t = t_
        
        # Move to the next (s)tate
        s = ToDoList.exec_action(s, a)
    
    return optimal_policy, t
