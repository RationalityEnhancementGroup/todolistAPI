import numpy as np

from collections import deque
from functools import reduce
from math import gcd
from tqdm import tqdm


def compute_gcd(goals):
    # Compute the GCD scale
    times = []
    
    for goal in goals:
        times.append(goal.get_effective_deadline())
        times.append(goal.get_unscheduled_time_est())
        
    gcd_scale = reduce(gcd, times)

    return gcd_scale


def compute_mixing_time(attainable_goals):
    """
    Computes mixing time between two consecutive (by deadline) attainable goals.

    Args:
        attainable_goals: List of attainable goals

    Returns:
        List of time available for mixing tasks between the current goal and
        future goals.
    """
    n = len(attainable_goals)  # Number of attainable goals
    mixing_time = np.zeros(shape=n, dtype=np.int32)
    
    # Initialize current time
    current_time_est = 0
    
    for goal_idx in range(n):
        
        # Get next goal from the sorted list of goals
        goal = attainable_goals[goal_idx]
        
        # Move to the end of the current goal
        current_time_est += goal.get_unscheduled_time_est()
        
        # Calculate time difference between two goal deadlines
        mixing_time[goal_idx] = goal.get_effective_deadline() - current_time_est
        
    return mixing_time


def compute_mixing_values(attainable_goals, mixing_parameter):
    mixing_values = np.zeros(len(attainable_goals))
    
    # Get value of the latest deadline
    max_deadline = attainable_goals[-1].get_effective_deadline()
    
    # Calculate distance between two consecutive goals
    for idx in range(len(attainable_goals) - 1):
        mixing_values[idx] = \
            attainable_goals[idx + 1].get_effective_deadline() \
            - attainable_goals[idx].get_effective_deadline()
        
    # Transform values s.t. the shortest distance has value == mixing_parameter
    mixing_values = (max_deadline - mixing_values) / max_deadline \
                    * mixing_parameter
    
    return mixing_values


def compute_optimal_values(goals, total_unscheduled_time_est, verbose=False):
    """
    Computes the maximum reward that can be attained by meeting the deadlines
    of the provided goals.
    
    Args:
        goals: [Goal]
        total_unscheduled_time_est: Total uncompleted time estimate for all
                                    tasks in the to-do list.
        verbose: TODO: ...

    Returns:
        Dynamic programming table of shape (number of goals + 1,
                                            latest_deadline + 1)
    """
    
    # Initialize constants
    d = total_unscheduled_time_est  # Set time horizon
    n = len(goals)  # Number of goals
    
    # Initialize dynamic programming table
    dp = np.zeros(shape=(n + 1, d + 1))
    
    if verbose:
        print('===== DP table shape =====')
        print(dp.shape, "\n")
        
        print('===== Goals =====')
        for goal in goals:
            print(f"{goal.get_description()} "
                  f"| Effective deadline: {goal.get_effective_deadline()}")
        print()

    # Compute the optimal values
    for i in tqdm(range(1, n + 1)):
        
        # Set goal index
        goal_idx = i - 1

        # Set estimated deadline of the current goal
        curr_est_deadline = goals[goal_idx].get_effective_deadline()
        
        for t in range(d + 1):
            
            # Get the latest possible time that we can schedule goal i
            t_ = min(t, curr_est_deadline) \
                - goals[goal_idx].get_unscheduled_time_est()
            
            if t_ < 0:
                dp[i, t] = dp[i - 1, t]
            else:
                d_i = goals[goal_idx].get_effective_deadline()
                dp[i, t] = max(dp[i - 1, t],
                               goals[goal_idx].get_reward(d_i) + dp[i - 1, t_])
                
    # Whether to print the optimal solution
    if verbose:
        print('===== DP table =====')
        print(dp, '\n')
        
    return dp


def generate_unattainable_msg(goal):
    tasks = goal.get_all_tasks()  # TODO: Store task number info in "cache"
    
    tp = "Goal"
    if len(tasks) == 1 and goal.get_description() == tasks[0].get_description():
        tp = "Task"
        
    return f'{tp} {goal.get_description()} is unattainable. Please extend ' \
           f'the deadline, remove inessential {"sub-" if tp == "Task" else ""}' \
           f'tasks or revise their time estimates, increase your working ' \
           f'hours, or drop this unachievable goal and prioritize other goals instead.'


def get_attainable_goals_dp(goals, dp):
    """
    Returns a list of attainable goals. If at least one goal is not attainable,
    it throws an error.
    
    Args:
        goals: [Goal]
        dp: Dynamic programming table

    Returns:
        Sorted list of attainable goals accompanied with their earliest starting
        time.
        - Example of an attainable-goals list: [(Goal, start_time), ...]
    """
    # Initialize parameters
    i = len(goals)  # Number of goals
    t = dp.shape[1] - 1  # Set horizon
    
    if t < 0:
        raise Exception("All goals have a negative deadline value!")
    
    # Initialize list of attainable goals
    attainable_goals = []

    # Get a sorted list of attainable
    while i != 0:
        goal_idx = i - 1
        
        if dp[i, t] == dp[i - 1, t]:
            i -= 1
            if (len(goals[goal_idx].get_unscheduled_tasks()) > 0 and
                goals[goal_idx].get_reward(t) >= 0):
                msg = generate_unattainable_msg(goals[goal_idx])
                raise Exception(msg)
            elif goals[goal_idx].get_reward(t) < 0:
                raise Exception(
                    f'Goal "{goals[goal_idx].get_description()}" '
                    f'has a negative reward value!')
        else:
            t_ = min(t, goals[goal_idx].effective_deadline) \
                 - goals[goal_idx].get_unscheduled_time_est()
            i -= 1
            t = t_
            
            attainable_goals = [goals[goal_idx]] + attainable_goals
        
        current_time_est = 0
        for goal in attainable_goals:
            goal_reward = goal.get_reward(current_time_est)
            
            for task in goal.get_all_tasks():
                task.set_reward(goal_reward)
                
            current_time_est += goal.get_unscheduled_time_est()
            
    return attainable_goals


def get_attainable_goals_greedy(goals):
    """
    Returns a list of attainable goals. If at least one goal is not attainable,
    it throws an error.
    
    Args:
        goals: List of goals.

    Returns:
        List of attainable goals.
    """
    # Initialize time
    current_time = 0
    
    attainable_goals = []
    
    for goal in tqdm(goals):
        next_time = current_time + goal.get_unscheduled_time_est()
        if next_time <= goal.get_effective_deadline():
            attainable_goals += [goal]
            current_time = next_time
        else:
            msg = generate_unattainable_msg(goal)
            raise Exception(msg)
        
    return attainable_goals
    

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
    # Initialize ordered task queue as all uncompleted tasks from the last goal
    ordered_tasks = deque(attainable_goals[-1].get_unscheduled_tasks())
    
    for goal_idx in tqdm(reversed(range(len(attainable_goals)-1))):
    
        # Initialize other task queue (Tasks that do not fit in available time)
        other_tasks = deque()
        
        # Get available time and mixing parameter for the current goal
        available_time = mixing_time[goal_idx]
        mixing_parameter = mixing_values[goal_idx]

        # Get all uncompleted unscheduled tasks from the current goal
        unscheduled_tasks = attainable_goals[goal_idx].get_unscheduled_tasks()
        
        # Convert tasks to list and sort w.r.t. deadline
        unscheduled_tasks = list(unscheduled_tasks)
        unscheduled_tasks.sort(key=lambda task: task.get_deadline())
        
        # Separate task with deadlines from other tasks in order to avoid
        # scheduling tasks after their deadline
        tasks_with_deadline = deque()
        batch_tasks = deque()
        
        for task in unscheduled_tasks:
            # If task has no "real" deadline
            if task.get_deadline_datetime() is None:
                batch_tasks.append(task)
            else:
                tasks_with_deadline.append(task)
        
        while len(ordered_tasks) > 0:
            task = ordered_tasks.popleft()
            task_time = task.get_time_est()

            if task_time <= available_time:
                batch_tasks.append(task)
                available_time -= task_time
            else:
                other_tasks.append(task)
                
            # If there is no time for other tasks, there is no point iterating
            if available_time == 0:
                other_tasks.extend(ordered_tasks)
                break

        # Mix tasks w.r.t. to the mixing parameter
        batch_tasks = shuffle(batch_tasks, mixing_parameter)

        # Append other tasks
        batch_tasks.extend(other_tasks)
        
        ordered_tasks = batch_tasks

    return ordered_tasks


def print_optimal_solution(goals, dp):
    """
    Prints optimal solution accompanied with the earliest start time of the
    attainable goals. It is used for debugging purposes only and the "error"
    messages are not shown to the end-user.
    
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
            t_ = min(t, goals[goal_idx].get_effective_deadline()) \
                 - goals[goal_idx].get_unscheduled_time_est()
            print_opt(i-1, t_)
            print(f'Attainable goal {goal_idx}!')
    
    latest_time = dp.shape[1] - 1
    
    print_opt(len(goals), latest_time)
    print()

    return


def scale_time(goals, scale, is_up=True):
    if scale > 1:
        
        for goal in goals:
            # Initialize new rewards dictionary
            new_rewards = dict()
            
            # Get reward for the latest deadline (assumption: 1 deadline/reward)
            reward = goal.get_reward(goal.get_latest_deadline_time())

            # Scale latest start time & uncompleted task time estimate
            goal.scale_est_deadline(scale, up=is_up)
            goal.scale_unscheduled_time_est(scale, up=is_up)
            
            # Set reward for the latest estimated deadline
            new_rewards[goal.effective_deadline] = reward
            
            # TODO: Extend to multiple deadlines and rewards!
            # Scale down goal deadline times
            # rewards = goal.get_reward_dict()
            #
            # for deadline, reward in rewards.items():
            #
            #     # Scaling up
            #     if is_up:
            #         deadline = deadline * scale
            #
            #     # Scaling down
            #     else:
            #         deadline = deadline // scale
            #
            #     new_rewards[deadline] = reward
            
            # Replace old rewards dictionary
            goal.set_rewards_dict(new_rewards)
            
    return goals


def shuffle(tasks_q, mixing_parameter=0.0):
    shuffled_q = deque()

    while len(tasks_q) > 0:
        task = tasks_q.popleft()
        
        if np.random.uniform() >= mixing_parameter:
            # Add task to the shuffled tasks queue
            shuffled_q.append(task)
            
        else:
            # Put task to the back of the queue for the next iteration
            tasks_q.append(task)
        
    return shuffled_q


def run_dp_algorithm(goals, verbose=False):
    """
    Computes maximum reward on a goal level by using dynamic programming.

    Source: http://www.cs.mun.ca/~kol/courses/2711-f13/dynprog.pdf [pages: 5-8]
    Complexity: O(nd + n log n)
              - d: Time of latest deadline
              - n: Number of goals

    Args:
        goals: [Goal]
        verbose: Whether to print DP table and optimal solution

    Returns:
        Sorted list of attainable goals accompanied with their earliest starting
        time.
        - Example of an attainable-goals list: [(Goal, start_time), ...]
    """
    # Compute GCD value
    gcd_scale = compute_gcd(goals)
    
    # gcd_scale = 1  # TODO: Remove!
    
    if verbose:
        print('===== GCD =====')
        print(gcd_scale, '\n')

    # Scale down time
    if gcd_scale > 1:
        goals = scale_time(goals, gcd_scale, is_up=False)

    # Compute total uncompleted time estimate
    total_unscheduled_time_est = \
        sum([goal.get_unscheduled_time_est() for goal in goals])

    # Compute optimal values
    dp = compute_optimal_values(goals, total_unscheduled_time_est, verbose)

    # Generate ordered lists of attainable and unattainable goals
    attainable_goals = get_attainable_goals_dp(goals, dp)

    # Scale up time
    if gcd_scale > 1:
        attainable_goals = scale_time(attainable_goals, gcd_scale, is_up=True)

    return attainable_goals


def run_greedy_algorithm(goals, verbose=False):
    """
    Computes maximum reward on a goal level by using a greedy algorithm.

    Complexity: O(n + n log n)
              - n: Number of goals

    Args:
        goals: [Goal]
        verbose: TODO: ...

    Returns:
        Sorted list of attainable goals accompanied with their earliest starting
        time.
        - Example of an attainable-goals list: [(Goal, start_time), ...]
    """
    return get_attainable_goals_greedy(goals)


def run_algorithm(to_do_list, algorithm_fn, mixing_parameter=.0, verbose=False):
    # Get list of goals
    goals = to_do_list.get_goals()
    
    # Sort goals in increasing order w.r.t. their (latest) deadlines
    goals.sort()
    
    # Get list of attainable goals
    attainable_goals = algorithm_fn(goals, verbose=verbose)
    
    # Compute mixing time
    mixing_time = compute_mixing_time(attainable_goals)
    
    # Compute mixing values
    mixing_values = compute_mixing_values(goals, mixing_parameter)

    # Get ordered task list (optimal sequence of tasks to complete)
    ordered_task_list = get_ordered_task_list(attainable_goals, mixing_time,
                                              mixing_values)
    
    if verbose:
        print('===== Attainable goals =====')
        if len(attainable_goals) == 0:
            print('None')
        else:
            for goal in attainable_goals:
                print(goal)

        print('===== Mixing time =====')
        print(mixing_time, '\n')

        print('===== Goal-mixing values =====')
        print(mixing_values, '\n')

        print('===== Ordered task list =====')
        for task in ordered_task_list:
            print(task)

    return ordered_task_list
