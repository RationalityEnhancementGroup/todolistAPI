import cherrypy
import numpy as np

from functools import reduce
from math import gcd
from tqdm import tqdm


def compute_gcd(goals):
    # Compute the GCD scale
    times = []
    
    for goal in goals:
        times.append(goal.get_latest_deadline_time())
        times.append(goal.get_uncompleted_time_est())
        
    gcd_scale = reduce(gcd, times)

    return gcd_scale


def compute_mixing_values(attainable_goals, mixing_parameter):
    mixing_values = np.zeros(len(attainable_goals))
    
    # Get value of the latest deadline
    max_deadline = attainable_goals[-1].get_latest_deadline_time()
    
    # Calculate distance between two consecutive goals
    for idx in range(len(attainable_goals) - 1):
        mixing_values[idx] = attainable_goals[idx + 1].get_latest_deadline_time() \
                             - attainable_goals[idx].get_latest_deadline_time()
        
    # Transform values s.t. the longest distance has value == mixing_parameter
    mixing_values = (max_deadline - mixing_values) / max_deadline \
                    * mixing_parameter
    
    return mixing_values


def compute_optimal_values(goals):
    """
    Computes the maximum reward that can be attained by meeting the deadlines
    of the provided goals.
    
    Args:
        goals: [Goal]

    Returns:
        Dynamic programming table of shape (number of goals + 1,
                                            latest_deadline + 1)
    """
    
    # Initialize constants
    d = goals[-1].get_latest_deadline_time()
    n = len(goals)  # Number of goals
    
    # Initialize dynamic programming table
    dp = np.zeros(shape=(n + 1, d + 1))
    
    # Compute the optimal values
    for i in tqdm(range(1, n + 1)):
        for t in range(d + 1):
            goal_idx = i - 1
            
            # Get the latest possible time that we can schedule goal i
            t_ = min(t, goals[goal_idx].get_latest_deadline_time()) \
                - goals[goal_idx].get_uncompleted_time_est()
            
            if t_ < 0:
                dp[i, t] = dp[i - 1, t]
            else:
                d_i = goals[goal_idx].get_latest_deadline_time()
                dp[i, t] = max(dp[i - 1, t],
                               goals[goal_idx].get_reward(d_i) + dp[i - 1, t_])
                
    return dp


def compute_simple_mixing_time(attainable_goals):
    """
    Computes mixing time between two consecutive (by deadline) attainable goals.

    Args:
        attainable_goals: List of attainable goals

    Returns:
        (mixing-time list, index of last 0 value - goal after which misc tasks
         can be completed)
    """
    n = len(attainable_goals)  # Number of attainable goals
    mixing_time = np.zeros(shape=n, dtype=np.int32)
    last_0_idx = 0  # The time when 0 was encountered
    
    current_time_est = 0
    for goal_idx in range(n):
        goal = attainable_goals[goal_idx]
        
        goal_latest_deadline = goal.get_latest_deadline_time()
        current_time_est += goal.get_uncompleted_time_est()
        
        mixing_time[goal_idx] = goal_latest_deadline - current_time_est
        
        if mixing_time[goal_idx] == 0:
            last_0_idx = goal_idx
        
    return mixing_time, last_0_idx


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
    t = goals[-1].get_latest_deadline_time()  # Latest deadline time
    
    # Initialize lists
    attainable_goals = []
    unattainable_goals = []

    # If the deadline of the latest goal is not in the future
    if t <= 0:
        unattainable_goals = goals
        
    else:
        # Get sorted lists of attainable and unattainable goals
        while i != 0:
            goal_idx = i - 1
            
            if dp[i, t] == dp[i - 1, t]:
                i -= 1
                unattainable_goals.append(goals[goal_idx])
            else:
                t_ = min(t, goals[goal_idx].get_latest_deadline_time()) \
                     - goals[goal_idx].get_uncompleted_time_est()
                i -= 1
                t = t_
                
                attainable_goals = [goals[goal_idx]] + attainable_goals
        
        attainable_goals.sort()
        current_time_est = 0
        for goal in attainable_goals:
            goal_reward = goal.get_reward(current_time_est)
            
            for task in goal.get_tasks():
                task.set_reward(goal_reward)
                
            current_time_est += goal.get_uncompleted_time_est()
            
        unattainable_goals.sort()
        
    return attainable_goals, unattainable_goals


def get_ordered_task_list(attainable_goals, mixing_time, mixing_values):
    """
    Mixes tasks from different goals in the reverse order so that tasks from
    distant goals can be assigned to be completed early.
    
    Args:
        attainable_goals: List of attainable goals.
        mixing_time: Array of available time for completing tasks from future
                     goals after the tasks from the i-th goal are completed.
                     Basically: goal.deadline - cumulative_time_est
        mixing_values: Probabilities of re-scheduling task in the future.

    Returns:
        Ordered list of tasks.
    """
    ordered_task_list = \
        list(attainable_goals[-1].get_uncompleted_tasks())

    for goal_idx in reversed(range(len(attainable_goals)-1)):
        batch_tasks = \
            list(attainable_goals[goal_idx].get_uncompleted_tasks())
        available_time = mixing_time[goal_idx]
        mixing_parameter = mixing_values[goal_idx]
        
        if available_time > 0:
            
            # TODO: Another mixing/acceptance parameter can be included here...
            for task in ordered_task_list:
                task_time = task.get_time_est()
                
                if task_time <= available_time:
                    batch_tasks += [task]
                    ordered_task_list.remove(task)
                    available_time -= task_time
                
                if available_time == 0:
                    break
            
            batch_tasks = shuffle(batch_tasks, mixing_parameter)
        
        ordered_task_list = batch_tasks + ordered_task_list

    return ordered_task_list


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
            t_ = min(t, goals[goal_idx].get_latest_deadline_time()) \
                 - goals[goal_idx].get_uncompleted_time_est()
            print_opt(i-1, t_)
            print(f'Attainable goal {goal_idx}!')
    
    latest_deadline = goals[-1].get_latest_deadline_time()
    
    if latest_deadline >= 0:
        print_opt(len(goals), latest_deadline)
        
    print()

    return


def scale_time(goals, scale, up=True):
    if scale > 1:
        
        for goal in goals:
            rewards = goal.get_reward_dict()
            new_rewards = dict()
            
            # Scale down goal deadline times
            for deadline, reward in rewards.items():
                
                # Scaling up
                if up:
                    deadline = deadline * scale
                    goal.scale_uncompleted_task_time(scale, up=True)
                    
                # Scaling down
                else:
                    deadline = deadline // scale
                    goal.scale_uncompleted_task_time(scale, up=False)
                    
                new_rewards[deadline] = reward
            
            # Replace old rewards dictionary
            goal.set_rewards_dict(new_rewards)
            
    return goals


def shuffle(tasks_list, mixing_parameter=0.0):
    shuffled_list = []

    # TODO: Take time estimation into account (?!)

    idx = 0
    while len(tasks_list) > 0:
        idx %= len(tasks_list)
        
        if np.random.uniform() >= mixing_parameter:
            
            # Add task to the shuffled tasks list
            shuffled_list += [tasks_list[idx]]
            
            # Remove task from the set of unshuffled tasks
            tasks_list = tasks_list[:idx] + tasks_list[idx+1:]

        else:
            idx += 1
        
    return shuffled_list


def simple_goal_scheduler(to_do_list, mixing_parameter=0.0, verbose=False):
    """
    Computes maximum reward on a goal level by using dynamic programming.

    Source: http://www.cs.mun.ca/~kol/courses/2711-f13/dynprog.pdf [pages: 5-8]
    Complexity: O(nd + n log n)
              - d: Time of latest deadline
              - n: Number of goals

    Args:
        to_do_list: ToDoList object
        mixing_parameter: Probability of skipping a task while mixing them.
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
    
    # Compute GCD value
    gcd_scale = compute_gcd(goals)
    
    # Scale down time
    if gcd_scale > 1:
        goals = scale_time(goals, gcd_scale, up=False)
    else:
        goals = goals
    
    # Compute optimal values
    dp = compute_optimal_values(goals)

    # Generate ordered lists of attainable and unattainable goals
    attainable_goals, unattainable_goals = get_attainable_goals(goals, dp)
    
    # Scale up time
    if gcd_scale > 1:
        goals = scale_time(goals, gcd_scale, up=True)

    if len(unattainable_goals) > 0:
        goals_str = ', '.join(goal.description for goal in unattainable_goals)
        raise Exception(f"Goals \"{goals_str[:-2]}\" are unattainable!")

    # Compute mixing time & mixing values
    mixing_time, last_0_idx = compute_simple_mixing_time(attainable_goals)
    mixing_values = compute_mixing_values(goals, mixing_parameter)

    # Get ordered task list (optimal sequence of tasks to complete)
    ordered_task_list = get_ordered_task_list(attainable_goals, mixing_time,
                                              mixing_values)

    # Whether to print the optimal solution
    if verbose:
        print('===== DP table =====')
        print(f"Shape: {dp.shape}", end="\n\n")
        print(dp, '\n')
    
        print('===== Attainable goals =====')
        if len(attainable_goals) == 0:
            print('None')
        else:
            for goal in attainable_goals:
                print(goal)
        print()
    
        print('===== Unattainable goals =====')
        if len(unattainable_goals) == 0:
            print('None')
        else:
            for goal in unattainable_goals:
                print(goal)
        print()
    
        print('===== Mixing time =====')
        print(mixing_time, '\n')
    
        print('===== Goal-mixing values =====')
        print(mixing_values, '\n')
        
        print('===== Ordered task list =====')
        for task in ordered_task_list:
            print(task)
        print()

    return ordered_task_list
