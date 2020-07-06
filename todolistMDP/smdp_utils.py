import numpy as np

from collections import deque
from todolistMDP.to_do_list import ToDoList, Goal


def compute_pseudo_rewards(obj, start_time=0, loc=0., scale=1):
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
                f = scale * f + loc + mean_goal_value
                
                # Get expected reward for (state, time, action)
                r = obj.R[s][t][a]["E"]
                
                # Store reward-shaping value
                obj.F[s][t][a]["E"] = f
                
                # Calculate pseudo-reward
                pr = f + r
                
                # Set pseudo-reward to 0 in case of numerical instability
                if np.isclose(pr, 0, atol=1e-6):
                    pr = 0
                
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
        item.set_optimal_reward(obj.PR[s][t][a]["E"])
        
        # Append (action, time') to the optimal-policy actions list
        optimal_policy.append({
            "s":   s,
            "a":   a,
            "t":   t,
            # "t_": t_,
            "obj": item,
            "PR":  obj.PR[s][t][a]["E"]
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
