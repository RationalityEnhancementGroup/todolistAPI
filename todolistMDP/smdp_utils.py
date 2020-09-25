import numpy as np

from collections import deque
from pprint import pprint
from todolistMDP.to_do_list import ToDoList


def compute_start_state_pseudo_rewards(to_do_list: ToDoList,
                                       bias=None, scale=None):
    
    # Get list of goals
    goals = to_do_list.get_goals()
    
    # Get goal future Q-values
    item_q = dict()  # {item ID: Q-value for item execution in s[0]}
    next_q = dict()  # {item ID: Q-value after item execution in s[0]}
    
    # Initialize best Q-value
    best_q = np.NINF
    
    # Initialize start time
    start_time = to_do_list.get_start_time()
    
    # Initialize list of incentivized items
    incentivized_items = deque()

    # Initialize total reward
    total_reward = 0
    
    # Initialize slack reward
    slack_reward = to_do_list.get_slack_reward()
    
    for goal in goals:
        
        if best_q <= slack_reward:
            best_q = slack_reward
        
        # Get time step after executing goal
        t_ = goal.get_time_est()
        
        # Set future Q-value
        future_q = goal.get_future_q(start_time + t_)
        
        # Compute Q-value in next step
        for item, q in goal.Q_s0.items():
            
            # Update Q-value with Q-values of future goals
            q += future_q
            
            # Get item ID
            item_id = item.get_id()

            # Store Q-value for item execution in s[0]
            item_q[item_id] = q
            
            # Get expected item reward
            reward = item.get_expected_reward()

            # Update total reward
            total_reward += reward

            # Compute Q-value after transition
            next_q[item_id] = q - reward
            
            # Update best Q-value and best next Q-value
            if best_q <= q:
                best_q = q

            # Add items to the list of incentivized items (?!)
            incentivized_items.append(item)
        
    # Initialize minimum pseudo-reward value (bias for linear transformation)
    min_pr = 0

    # Initialize sum of pseudo-rewards
    sum_pr = 0

    # Compute untransformed pseudo-rewards
    for item in incentivized_items:
        
        # Get item ID
        item_id = item.get_id()
        
        # Compute pseudo-reward
        pr = next_q[item_id] - best_q
        
        if np.isclose(pr, 0, atol=1e-6):
            pr = 0

        # Store pseudo-reward
        item.set_optimal_reward(pr)
        
        # Update minimum pseudo-reward
        min_pr = min(min_pr, pr)
        
        # Update sum of pseudo-rewards
        sum_pr += pr

    # Compute sum of goal values
    sum_goal_values = sum([goal.get_reward() for goal in goals])
    
    # Set value of scaling parameter
    if scale is None:
        
        # As defined in the report
        scale = 1.10

    # Set value of bias parameter
    if bias is None:
        
        # Total number of incentivized items
        n = len(incentivized_items)
        
        # Derive value of the bias term
        bias = (sum_goal_values - scale * sum_pr) / n
        
        # Take total reward into account
        bias -= (total_reward / n)
        
    print("Bias:", bias)
    print("Scale:", scale)
    print()

    # Initialize {item ID: pseudo-reward} dictionary
    id2pr = dict()

    # Sanity check for the sum of pseudo-rewards
    sc_sum_pr = 0

    # Perform linear transformation on item pseudo-rewards
    for item in incentivized_items:
        
        # Get item unique identification
        item_id = item.get_id()
    
        # Transform pseudo-reward
        pr = f = scale * item.get_optimal_reward() + bias
    
        # Get expected item reward
        item_reward = item.get_expected_reward()

        # Add immediate reward to the pseudo-reward
        pr += item_reward
        
        # print(
        #     f"{item.get_description():<70s} | "
        #     # f"{best_next_q:>8.2f} | "
        #     f"max Q*(s', a'): {next_q[item_id]:>8.2f} | "
        #     f"Q*(s, a): {item_q[item_id]:>8.2f} | "
        #     f"V*(s): {best_q:>8.2f} | "
        #     f"f*(s, a): {item.get_optimal_reward():8.2f} | "
        #     f"f*(s, a) + b: {f:8.2f} | "
        #     f"r(s, a, s'): {item.get_expected_reward():>8.2f} | "
        #     f"r'(s, a, s'): {pr:>8.2f}"
        # )
    
        # Store new (zero) pseudo-reward
        item.set_optimal_reward(pr)
    
        # If item is not slack action
        if item.get_idx() != -1:
            
            # Update sanity check for the sum of pseudo-rewards
            sc_sum_pr += item.get_optimal_reward()
    
        # Store pseudo-reward {item ID: pseudo-reward}
        id2pr[item_id] = pr

    # print(f"\nTotal sum of pseudo-rewards: {sc_sum_pr:.2f}\n")
    
    return {
        "incentivized_items": incentivized_items,
        
        "id2pr": id2pr,
        "sc_sum_pr": sc_sum_pr,
        "scale": scale,
        "bias": bias
    }
