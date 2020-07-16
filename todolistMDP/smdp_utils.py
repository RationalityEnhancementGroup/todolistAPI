import numpy as np

from collections import deque
from pprint import pprint
from todolistMDP.to_do_list import ToDoList, Goal


def compute_pseudo_rewards(obj, start_time=0, loc=0., scale=1.):
    """
    Computes pseudo-rewards.

    Args:
        obj: Object for which pseudo-rewards are computed. Eg. Goal or ToDoList
        start_time: Initial SMDP time.
        loc: Bias parameter of the linear transformation.
        scale: Scaling parameter of the linear transformation.

    Returns:
        /
    """
    standardizing_reward = obj.highest_negative_reward
    if obj.highest_negative_reward == np.NINF:
        standardizing_reward = 0
    
    # Initialize minimum pseudo-reward
    min_PR = np.PINF
    
    # Get time-shift discount
    discount = ToDoList.get_discount(start_time)
    
    for s in obj.Q.keys():
        
        for t in obj.Q[s].keys():
            
            # The best possible (a)ction and (q)-value in state s
            best_a, best_q = ToDoList.max_from_dict(obj.Q[s][t])
            best_q *= discount
            
            # Update optimal policy
            obj.P[s][t] = best_a
            
            for a in obj.Q[s][t].keys():
                
                # TODO: Move to `solve` function(s)
                obj.PR[s][t].setdefault(a, dict())
                obj.F[s][t].setdefault(a, dict())
                
                # Initialize action object
                action_obj = None
                
                # Initialize mean goal value
                mean_goal_value = 0
                
                if a is None:
                    v_ = 0
                
                elif a == -1:
                    action_obj = obj.get_slack_action()
                    v_ = obj.Q[s][t][a]["E"]
                
                else:
                    if type(obj) is Goal:
                        
                        # Get current Task object
                        action_obj = obj.tasks[a]
                        
                        # Move to the next state
                        s_ = ToDoList.exec_action(s, a)
                        
                        # TODO: Compute expected future reward as a function
                        if "E" not in obj.Q[s_].keys():
                            pass
                        
                        # Get time transitions to the next state
                        time_transitions = action_obj.get_time_transitions()
                        
                        # Initialize expected state value
                        v_ = 0
                        
                        prop_goal_values = deque()
                        mean_goal_value_scale = 0
                        
                        for time_est, prob_t_ in time_transitions.items():
                            
                            # Make time transition
                            t_ = t + time_est
                            
                            # Compute mean goal value
                            prop_goal_values.extend([
                                [goal.get_reward(
                                    start_time) / goal.get_time_est()
                                 for goal in action_obj.get_goals()]
                            ])
                            
                            # Get gamma for the next transition
                            gamma = ToDoList.get_discount(time_est)
                            
                            # Get optimal action and value in the next state
                            a_, q_ = ToDoList.max_from_dict(obj.Q[s_][t_])
                            
                            # Update expected value of the next state
                            v_ += prob_t_ * gamma * q_
                            
                            # Get correct Q-value for the terminal state
                            if a_ is None:
                                v_ = best_q
                                if best_a != -1:
                                    v_ -= obj.R[s][t][a]["E"]
                            
                            mean_goal_value_scale += time_est * prob_t_
                        
                        #
                        mean_goal_value = np.mean(prop_goal_values)
                        mean_goal_value *= mean_goal_value_scale
                    
                    elif type(obj) is ToDoList:
                        
                        # Get current Goal object
                        action_obj = obj.goals[a]
                        
                        # Move to the next state
                        s_ = ToDoList.exec_action(s, a)
                        
                        # Get expected goal time estimate
                        time_est = action_obj.get_time_est()
                        
                        # Make time transition
                        t_ = t + time_est
                        
                        # Get gamma for the next transition
                        gamma = ToDoList.get_discount(time_est)
                        
                        # Get optimal action in the next state
                        a_, v_ = ToDoList.max_from_dict(obj.Q[s_][t_])
                        
                        # Compute discounted action value
                        v_ *= gamma
                    
                    else:
                        raise NotImplementedError(
                            f"Unknown object type {type(obj)}")
                
                # Expected Q-value of the next state
                v_ *= discount
                
                # Compute value of the reward-shaping function
                f = v_ - best_q
                
                # Standardize rewards s.t. negative rewards <= 0
                # f -= standardizing_reward * discount
                
                # Make affine transformation of the reward-shaping function
                f = scale * f + loc  # + mean_goal_value
                
                # Get expected reward for (state, time, action)
                r = obj.R[s][t][a]["E"]
                
                # Store reward-shaping value
                obj.F[s][t][a]["E"] = f
                
                # Calculate pseudo-reward
                pr = f + r
                
                # Set pseudo-reward to 0 in case of numerical instability
                if np.isclose(pr, 0, atol=1e-6):
                    pr = 0.
                
                # Store pseudo-reward
                obj.PR[s][t][a]["E"] = pr
                
                # Store minimum non-infinite pseudo-reward
                if action_obj is not None and (pr != np.NINF or pr != np.PINF):
                    min_PR = min(min_PR, pr)
                
                # Add PR for each occurrence of action
                if action_obj is not None and t != np.PINF:
                    action_obj.values.append(pr)
                    
                    # if type(obj) is ToDoList:
                    #     # action_obj.values.append(
                    #     #     pr * np.exp(obj.log_prob[s][t])
                    #     # )
                    #     action_obj.values.append(
                    #         v_ * np.exp(obj.log_prob[s][t])
                    #     )
                    #
                    # elif type(obj) is Goal:
                    #     # action_obj.values.append(
                    #     #     pr * np.exp(obj.log_prob[s][t])
                    #     # )
                    #     action_obj.values.append(
                    #         v_ * np.exp(obj.log_prob[s][t])
                    #     )
                    #
                    # else:
                    #     pass
    
    return min_PR


def compute_s0_pseudo_rewards(to_do_list):
    
    # Get list of goals
    goals = to_do_list.get_goals()
    
    # Get goal future Q-values
    # future_q = dict()  # {Goal index: Future value after exec in s[0]}
    task_q = dict()  # {Task ID: Q-value for task execution in s[0]}
    next_q = dict()  # {Task ID: Q-value after task execution in s[0]}
    
    # Initialize best Q-value
    best_q = np.NINF
    best_next_q = None
    # TODO: Keep track of the best next action
    
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
        s = tuple(0 for _ in range(goal.get_num_tasks()))
        t = 0
        
        tasks = goal.get_tasks()
        
        for a in goal.get_q_values(s, t):
            
            if a != -1:
                task = tasks[a]
                
                # Get task ID
                task_id = task.get_id()
                
                # Compute task Q-value
                q = goal.get_q_values(s, t, a) + future_q
                
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
    
                # best_next_q = max(best_next_q, q - loss)
    
                # Add tasks to the list of incentivized tasks (?!)
                incentivized_tasks.append(task)
                
            else:
                pass
                # TODO: Enable slack action
                
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
    scale = sum_goal_values / sum_pr
    bias = (1 - min_pr) * scale

    # Initialize {Task ID: pseudo-reward} dictionary
    id2pr = dict()

    # Get total number of tasks
    # num_tasks = len(optimal_tasks) + len(suboptimal_tasks)

    # Sanity check for the sum of pseudo-rewards
    sc_sum_pr = 0

    # Perform linear transformation on task pseudo-rewards
    # for task in optimal_tasks + suboptimal_tasks + slack_tasks:
    for task in incentivized_tasks:
        
        # TODO: Include slack actions...
    
        # Get task unique identification
        task_id = task.get_id()
    
        # Transform pseudo-reward
        pr = scale * task.get_optimal_reward() + bias
    
        # Store new (zero) pseudo-reward
        task.set_optimal_reward(pr)
    
        # If task is not slack action
        if task.get_idx() is not None:
            
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
            
            # Compute 0-state pseudo-rewards for current goal
            PR, best_a = goal.get_s0_pseudo_rewards(t)
            
            # Get all goal's tasks
            tasks = goal.get_tasks()
            
            for task in goal.sorted_tasks_by_deadlines:
                
                # Get task ID
                task_id = task.get_id()
                
                # Get task index
                task_idx = task.get_idx()
                
                # Get task object for the corresponding index
                task = tasks[task_idx]

                # TODO: Comment
                if task_id not in id2pr.keys():
                    break

                # Append task to the queue of tasks to be scheduled
                if best_a != -1:
                    optimal_tasks.append(task)
                else:
                    suboptimal_tasks.append(task)
                    
                # Set transformed task pseudo-reward as optimal value
                task.set_optimal_reward(id2pr[task_id])
                
            # If the goal is worth pursuing (i.e. slack action is not the best)
            if best_a == -1:

                # TODO: Slack-action ID... (?)
                
                # Get slack action associated with current goal
                slack_action = goal.get_slack_action()
                
                # Set slack action name
                # TODO: Add goal name to slack-off action
                slack_action.set_description(
                    f"Please revise goal \"{goal.get_description()}\"!"
                )
                
                # Add slack action to the list of slack tasks
                slack_tasks.append(goal.get_slack_action())
        
        # TODO: Add slack-off action that reminds user to revise other goals
        #     - Not sure if necessary, i.e. the method reaches this point...
        else:
            pass
        
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
            time_est = np.random.choice(times, size=1, replace=False, p=values)[
                0]
        
        else:
            raise NotImplementedError(f"Unsupported choice mode {choice_mode}!")
        
        # Compute next time transition
        t_ = t + time_est
        
        # Move to the next time step
        t = t_
        
        # Move to the next (s)tate
        s = ToDoList.exec_action(s, a)
    
    return optimal_policy, t
