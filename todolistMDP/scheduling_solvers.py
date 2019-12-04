import numpy as np


def compute_optimal_values(goals):
    """
    Computes the maximum reward that can be atttained by meeting the deadlines
    of the provided goals.
    
    Args:
        goals: [Goal]

    Returns:
        Dynamic programming table of shape (number of goals + 1,
                                            latest_deadline + 1)
    """
    # Initialize constants
    d = goals[-1].get_deadline_time()  # Latest deadline time
    n = len(goals)  # Number of goals

    # Initialize dynamic programming table
    dp = np.zeros(shape=(n + 1, d + 1), dtype=np.int32)
    
    # Compute the optimal values
    for i in range(1, n + 1):
        for t in range(d + 1):
            goal_idx = i - 1
            
            # Get the latest possible time that we can schedule goal i
            t_ = min(t, goals[goal_idx].get_deadline_time()) \
                - goals[goal_idx].get_total_time_est()
            
            if t_ < 0:
                dp[i, t] = dp[i - 1, t]
            else:
                d_i = goals[goal_idx].get_deadline_time()
                dp[i, t] = max(dp[i - 1, t],
                               goals[goal_idx].get_reward(d_i) + dp[i - 1, t_])
                
    return dp


def get_attainable_goals(goals, dp):
    """
    Splits the set of goals into attainable and unattainable goals.
    
    Args:
        goals: [Goal]
        dp: Dynamic programming table

    Returns:
        Sorted list of attainable goals accompanied with their earliest starting
        time and a list of unattainable goals.
        - Example of an attainable-goals list: [(Goal, start_time), ...]
        - Example of an unattainable-goals list: [Goal, Goal, ...]
    """
    # Initialize parameters
    i = len(goals)  # Number of goals
    t = goals[-1].get_deadline_time()  # Latest deadline time
    
    # Initialize lists
    attainable_goals = []
    unattainable_goals = []
    
    # Get sorted lists of attainable and unattainable goals
    while i != 0:
        goal_idx = i - 1
        
        if dp[i, t] == dp[i - 1, t]:
            i -= 1
            unattainable_goals.append(goals[goal_idx])
        else:
            t_ = min(t, goals[goal_idx].get_deadline_time()) \
                 - goals[goal_idx].get_total_time_est()
            i -= 1
            t = t_
            
            attainable_goals.append((goals[goal_idx], t_))
    
    attainable_goals.sort()
    unattainable_goals.sort()
    
    if len(attainable_goals) == 0:
        attainable_goals = [None]
        
    if len(unattainable_goals) == 0:
        unattainable_goals = [None]

    return attainable_goals, unattainable_goals


def print_optimal_solution(goals, dp):
    """
    Prints optimal solution accompanied with the earliest start time of the
    attainable goals.
    
    Args:
        goals: [Goals]
        dp: Dynamic programming table

    Returns:
        /
    """

    def print_opt(i, t):
        if i == 0:
            return None
    
        goal_idx = i-1
    
        if dp[i, t] == dp[i-1, t]:
            print_opt(i-1, t)
            print(f'Unattainable goal {goal_idx}!')
        else:
            t_ = min(t, goals[goal_idx].get_deadline_time()) \
                 - goals[goal_idx].get_total_time_est()
            print_opt(i-1, t_)
            print(f'Attainable goal {goal_idx} at time {t_}')
    
    print_opt(len(goals), goals[-1].get_deadline_time())
    print()

    return


def simple_goal_scheduler(to_do_list, verbose=False):
    """
    Computes maximum reward on a goal level by using dynamic programming.

    Source: http://www.cs.mun.ca/~kol/courses/2711-f13/dynprog.pdf [pages: 5-8]
    Complexity: O(nd + n log n)
              - d: Time of latest deadline
              - n: Number of goals

    Args:
        to_do_list: ToDoList object
        verbose: Whether to print DP table and optimal solution

    Returns:
        Sorted list of attainable goals accompanied with their earliest starting
        time and a list of unattainable goals.
        - Example of an attainable-goals list: [(Goal, start_time), ...]
        - Example of an unattainable-goals list: [Goal, Goal, ...]
    """
    # Get list of goals
    goals = to_do_list.get_goals()
    
    # Sort goals in increasing order w.r.t. their (latest) deadlines
    goals.sort()
    
    # Compute optimal values
    dp = compute_optimal_values(goals)
    
    # Generate ordered lists of attainable and unattainable goals
    attainable_goals, unattainable_goals = get_attainable_goals(goals, dp)
    
    # Whether to print the optimal solution
    if verbose:
        print(dp, '\n')
        print_optimal_solution(goals, dp)
        
        print('===== Attainable goals =====')
        for goal, time in attainable_goals:
            print(f'Earliest start time: {time}')
            print(goal)
        
        print('===== Unattainable goals =====')
        for goal in unattainable_goals:
            print(goal)
    
    return attainable_goals, unattainable_goals
