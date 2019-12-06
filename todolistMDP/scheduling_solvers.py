import numpy as np


def compute_mixing_time(attainable_goals):
    """
    Explanation for the table:
    The i-th element of the main diagonal is the amount of time between the
    deadline of the i-th goal and the sum of time estimations of the preceding
    goals and the time estimation of the i-th goal.
    diag(i) = goal[i].deadline - sum(goal[:i].time_est) (inclusive interval)
    
    The row i and column j (where i < j) of the table denotes the available time
    to complete any task from the goals [i+1:j] (inclusive interval). If the
    value is 0, then the tasks of the i-th goal cannot be mixed with any of the
    future [i+1:j]-th goals. Otherwise (if the value is not 0), then there is a
    possibility to mix tasks of the future [i+1:j] goals with tasks of the i-th
    goal with available time slot equal to the value of the [i, j]-th element
    of the table.
    
    If the [0, (n-1)]-th element of the table is 0. Then no miscellaneous tasks
    can be scheduled! Otherwise, the last column of the table indicates the time
    available for miscellaneous tasks after the i-th goal (row index).
    
    Example:
        [3 2 0 0]
        [0 2 0 0]
        [0 0 0 0]
        [0 0 0 2]
        
        - The [0, 1]-th element (value = 2) of the table says that there are 2
        time units (minutes) available to use for tasks from the 1-st goal.
        - The [0, 3]-th element (value = 0) of the table says that there is no
        time available to use for tasks from the 1-st, 2-nd and 3-rd goal
        because there is no time to perform any task from the 3-rd goal before
        the 2-nd goal is completed.
    
    Args:
        attainable_goals: List of attainable goals

    Returns:
        (mixing-time table, index of last 0)
    """
    n = len(attainable_goals)  # Number of attainable goals
    mixing_time = np.zeros(shape=(n, n), dtype=np.int32)
    last_0_idx = 0  # The time when 0 was encountered
    
    current_time_est = 0
    for goal_idx in range(n):
        goal = attainable_goals[goal_idx]
        
        latest_deadline = goal.get_deadline_time()
        current_time_est += goal.get_uncompleted_time_est()
        
        mixing_time[goal_idx, goal_idx] = latest_deadline - current_time_est
        
        if mixing_time[goal_idx, goal_idx] == 0:
            last_0_idx = goal_idx
    
    for from_idx in range(n):
        for until_idx in range(from_idx + 1, n):
            mixing_time[from_idx, until_idx] = \
                min(mixing_time[from_idx, until_idx - 1],
                    mixing_time[from_idx + 1, until_idx])
            
    return mixing_time, last_0_idx


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
        
        # TODO: Maybe return [Task]?!
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
            goals[goal_idx].set_not_attainable()
        else:
            t_ = min(t, goals[goal_idx].get_deadline_time()) \
                 - goals[goal_idx].get_total_time_est()
            i -= 1
            t = t_
            
            attainable_goals.append(goals[goal_idx])
            goals[goal_idx].set_attainable(t_)
    
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
            print(f'Attainable goal {goal_idx} at time '
                  f'{goals[goal_idx].get_earliest_start_time()}')
    
    print_opt(len(goals), goals[-1].get_deadline_time())
    print()

    return


def simple_goal_scheduler(to_do_list, verbose=False, mix_goals=False):
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

    # Compute mixing time
    mixing_time, last_0_idx = compute_mixing_time(attainable_goals)
    
    # Whether to print the optimal solution
    if verbose:
        print('===== DP table =====')
        print(dp, '\n')
        print_optimal_solution(goals, dp)
        
        print('===== Attainable goals =====')
        for goal in attainable_goals:
            print(goal)
        
        print('===== Unattainable goals =====')
        for goal in unattainable_goals:
            print(goal)
        print()

        print('===== Mixing time =====')
        print(mixing_time, '\n')

        if mixing_time[-1, -1] == 0:
            print('No time for miscellaneous tasks!', end='\n\n')
        else:
            print(f'There is time for miscellaneous tasks after the deadline of'
                  f' goal {last_0_idx}!', end='\n\n')

    return attainable_goals, unattainable_goals, mixing_time
