from copy import deepcopy
from src.point_scalers import scale_optimal_rewards
from src.utils import tree_to_old_structure
from src.utils import separate_tasks_with_deadlines, task_list_from_projects
from src.schedulers.schedulers import schedule_tasks_for_today

from todolistMDP.scheduling_solvers import run_algorithm
from todolistMDP.to_do_list import *
from todolistMDP.test_smdp import d_bm, print_item, print_stats


def assign_constant_points(projects, default_task_value=10):
    """
    Takes in parsed project tree, with one level of tasks
    Outputs project tree with constant points assigned
    """
    for goal in projects:
        for task in goal["ch"]:
            task["val"] = default_task_value
    return projects


def assign_dynamic_programming_points(projects, solver_fn,
                                      scaling_fn, scaling_inputs,
                                      day_duration=8 * 60, **params):
    
    # Separate tasks with deadlines from real goals
    # goals = separate_tasks_with_deadlines(deepcopy(projects))

    # Convert real goals from JSON to Goal class objects
    goals = tree_to_old_structure(projects)
    # goals = tree_to_old_structure(goals)
    
    # Add them together into a single list
    to_do_list = ToDoList(goals, start_time=0)

    # Get ordered list of tasks
    ordered_tasks = \
        run_algorithm(to_do_list, solver_fn,
                      mixing_parameter=params["mixing_parameter"],
                      verbose=True)

    # Get all tasks to be scheduled today
    tasks = deque()
    for goal in goals:
        tasks.extend(goal.get_scheduled_tasks())

    # Add additional tasks to be scheduled
    # tasks.extend(ordered_tasks)

    # Scale task values according to the provided scaling function
    scaling_fn(tasks, **scaling_inputs)
    
    # Schedule tasks for today
    today_tasks = schedule_tasks_for_today(projects, tasks,
                                           duration_remaining=day_duration,
                                           time_zone=params['time_zone'])

    return today_tasks  # List of tasks from projects


def assign_length_points(projects):
    """
    Takes in parsed and flattened project tree
    Outputs project tree with points assigned according to length heuristic
    """
    for goal in projects:
        value_per_minute = goal["value"]/float(sum([child["est"]
                                                    for child in goal["ch"]]))
        for child in goal["ch"]:
            child["val"] = child["est"]/float(value_per_minute)
    return projects


def assign_old_api_points(projects, solver_fn, duration=8*60, **params):
    """
    input: parsed project tree, duration of current day, and planning function
    output: task list of tasks to be done that day (c.f. schedulers.py)

    The provided solver function creates an ToDoListMDP object and solves the
    MDP. Possible solver functions (so far):
        - backward_induction
        - policy_iteration
        - simple_goal_scheduler
        - value_iteration
    
    **params:
        - mixing_parameter [0, 1): Probability of delaying a task in scheduling
    """
    old_structure = tree_to_old_structure(projects)
    to_do_list = ToDoList(old_structure, start_time=0)
    mdp = solver_fn(to_do_list)
    mdp.scale_rewards()

    actions_and_rewards = []
    task_list = task_list_from_projects(projects)
    tasks = mdp.to_do_list.get_tasks()
    today_tasks = [task["id"] for task in task_list if task["today"] == 1]

    state = (tuple(0 for task in mdp.get_tasks_list()), 0)  # Starting state
    # first schedule today tasks
    for today_task in today_tasks:
        action = next(item for item in mdp.get_possible_actions(state)
                      if (today_task == tasks[item].description))
        next_state_and_prob = mdp.get_trans_states_and_probs(state, action)
        next_state = next_state_and_prob[0][0]
        duration -= next_state[1]
        reward = mdp.get_expected_pseudo_rewards(state, action,
                                                 transformed=False)
        state = next_state
        actions_and_rewards.append((action, reward))

    # Then schedule based on mdp
    still_scheduling = True
    while still_scheduling:
        still_scheduling = False
        possible_actions = mdp.get_possible_actions(state)
        q_values = [mdp.get_q_value(state, action, mdp.V_states)
                    for action in possible_actions]
        # Go through possible actions in order of q values
        for action_index in np.argsort(q_values)[::-1]:
            possible_action = possible_actions[action_index]
            next_state_and_prob = \
                mdp.get_trans_states_and_probs(state, possible_action)
            next_state = next_state_and_prob[0][0]

            # See if the next best action would fit into that day's duration
            if (duration-next_state[1]) >= 0:
                duration -= next_state[1]
                reward = mdp.get_expected_pseudo_rewards(state, possible_action,
                                                         transformed=False)
                state = next_state
                still_scheduling = True
                actions_and_rewards.append((possible_action, reward))
                break

    final_tasks = []
    for action, reward in actions_and_rewards:
        final_tasks.append((next(item for item in task_list
                                 if (item["id"] == tasks[action].description))))
        final_tasks[-1]["val"] = reward

    return final_tasks


def assign_random_points(projects, distribution_fxn=np.random.normal,
                         fxn_args=(10, 2), min_value=1.,
                         max_value=float('inf')):
    """
    Takes in parsed project tree, with one level of tasks
    Outputs project tree with random points assigned according to distribution
    function with inputted args
    """
    for goal in projects:
        for task in goal["ch"]:
            # Bound values in the interval [min_value, max_value]
            task["val"] = max(min_value,
                              min(max_value, distribution_fxn(*fxn_args)))
            
            # In case of negative rewards, convert them to absolute value
            # - This also depends on the choice of lower bound
            task["val"] = abs(task["val"])
    
    return projects


def assign_smdp_points(projects, day_duration, smdp_params, timer,
                       json=True, start_time=0, time_zone=0, verbose=False):
    
    # Separate tasks with deadlines from real goals
    # goals = separate_tasks_with_deadlines(deepcopy(projects))
    
    if json:
        
        # Start timer: Run SMDP - Converting JSON to objects
        tic = time.time()

        # Convert real goals from JSON to Goal class objects
        goals = tree_to_old_structure(projects, smdp_params)

        # Stop timer: Run SMDP - Converting JSON to objects
        toc = time.time()

        # Store time: Run SMDP - Converting JSON to objects
        timer["Run SMDP - Converting JSON to objects"] = toc - tic

    else:
        goals = projects

    # Start timer: Run SMDP - Creating ToDoList object
    tic = time.time()

    # Add them together into a single list
    to_do_list = ToDoList(goals,
                          gamma=smdp_params["gamma"],
                          slack_reward=smdp_params["slack_reward"],
                          start_time=start_time)
    
    # Stop timer: Run SMDP - Creating ToDoList object
    toc = time.time()

    # Store timer: Run SMDP - Creating ToDoList object
    timer["Run SMDP - Creating ToDoList object"] = toc - tic

    # Start timer: Run SMDP - Solving SMDP
    tic = time.time()

    # Solve to-do list
    to_do_list.solve(verbose=verbose)

    # Stop timer: Run SMDP - Solving SMDP
    toc = time.time()

    # Store time: Run SMDP - Solving SMDP
    timer["Run SMDP - Solving SMDP"] = toc - tic

    # Start timer: Run SMDP - Computing goal-level pseudo-rewards
    tic = time.time()

    # Compute goal-level pseudo-rewards
    compute_pseudo_rewards(to_do_list, start_time=start_time,
                           loc=smdp_params["goal_pr_loc"],
                           scale=smdp_params["goal_pr_scale"])
    
    # Stop timer: Run SMDP - Computing goal-level pseudo-rewards
    toc = time.time()

    # Store time: Run SMDP - Computing goal-level pseudo-rewards
    timer["Run SMDP - Computing goal-level pseudo-rewards"] = toc - tic

    if verbose:
        print_item(to_do_list)
    
    # Start timer: Run SMDP - Running goal-level optimal policy
    tic = time.time()

    # Get optimal goal-level policy
    P, t = run_optimal_policy(to_do_list,
                              choice_mode=smdp_params["choice_mode"])
    
    # Stop timer: Run SMDP - Running goal-level optimal policy
    toc = time.time()

    # Store time: Run SMDP - Running goal-level optimal policy
    timer["Run SMDP - Running goal-level optimal policy"] = toc - tic

    # Initialize list of all tasks
    tasks = deque()
    
    # Start timer: Run SMDP - Task-level optimal policy & pseudo-rewards
    tic = time.time()

    for entry in P:
        
        # Get next (a)ction and initial (t)ime
        a = entry["a"]
        t = entry["t"]

        # If next action is not slack-off action
        if a != -1:
            
            # Get goal that correspond to that (a)ction
            goal = goals[a]
            
            # Compute pseudo-rewards at initial (t)time
            compute_pseudo_rewards(goal, start_time=start_time,
                                   loc=smdp_params["task_pr_loc"],
                                   scale=smdp_params["task_pr_scale"])
            
            # Run optimal policy at initial (t)ime
            P_, t = run_optimal_policy(goal, t=t,
                                       choice_mode=smdp_params["choice_mode"])
            
            # Add tasks to the list of all tasks
            tasks.extend(P_)
            
            if verbose:
                print_item(goal)
                
    # Stop timer: Run SMDP - Task-level optimal policy & pseudo-rewards
    toc = time.time()
    
    # Store time: Run SMDP - Task-level optimal policy & pseudo-rewards
    timer["Run SMDP - Task-level optimal policy & pseudo-rewards"] = toc - tic

    # Start timer: Run SMDP - Converting tasks queue to list
    tic = time.time()

    # Convert tasks queue to list
    tasks = [task["obj"] for task in tasks]

    # Stop timer: Run SMDP - Converting tasks queue to list
    toc = time.time()

    # Store time: Run SMDP - Converting tasks queue to list
    timer["Run SMDP - Task-level optimal policy & pseudo-rewards"] = toc - tic

    # Start timer: Run SMDP - Sorting tasks w.r.t. optimal reward
    tic = time.time()

    # Sort tasks w.r.t. optimal reward (pseudo-reward)
    tasks.sort(key=lambda task: -task.get_optimal_reward())
    
    # Stop timer: Run SMDP - Sorting tasks w.r.t. optimal reward
    toc = time.time()

    # Store time: Run SMDP - Sorting tasks w.r.t. optimal reward
    timer["Sorting tasks w.r.t. optimal reward"] = toc - tic

    if verbose:
        for task in tasks:
            print(task.get_id(), task.get_optimal_reward())

    # Start timer: Run SMDP - Scaling rewards
    tic = time.time()

    # Scale task values according to the provided scaling function
    scale_optimal_rewards(tasks,
                          scale_min=smdp_params["scale_min"],
                          scale_max=smdp_params["scale_max"],
                          scale_type=smdp_params["scale_type"])
    
    # Stop timer: Run SMDP - Scaling rewards
    toc = time.time()

    # Store time: Run SMDP - Scaling rewards
    timer["Run SMDP - Scaling rewards"] = toc - tic

    if json:
        
        # Start timer: Run SMDP - Scheduling tasks
        tic = time.time()

        # Schedule tasks for today
        today_tasks = schedule_tasks_for_today(projects, tasks,
                                               duration_remaining=day_duration,
                                               time_zone=time_zone)
        
        # Stop timer: Run SMDP - Scheduling tasks
        toc = time.time()
        
        # Start timer: Run SMDP - Scheduling tasks
        timer["Run SMDP - Scheduling tasks"] = toc - tic

    else:
        today_tasks = tasks
        
    return today_tasks


def get_actions_and_rewards(mdp, verbose=False):
    policy = mdp.get_optimal_policy()
    values = mdp.get_value_function()
    
    state = sorted(list(policy.keys()))[0]
    vec, tm = state
    
    action = policy[state]
    value = values[state]
    actions_and_rewards = [(action, value)]

    if verbose:
        print(state, action, value)
    
    # While there is an action to perform
    while policy[state] is not None:
        
        # Get task
        idx = policy[state]
        task = mdp.index_to_task[idx]
        
        # Get next state
        next_vec = list(vec)
        next_vec[idx] = 1
        
        tm += task.get_time_est()
        state = (tuple(next_vec), tm)
        vec, tm = state
        
        action = policy[state]
        value = values[state]
    
        if verbose:
            print(state, action, value)
        actions_and_rewards += [(action, value)]
    
    return actions_and_rewards


if __name__ == '__main__':
    choice_mode = "max"
    gamma = 0.99
    goal_pr_loc = 0
    goal_pr_scale = 1.
    task_pr_loc = 0
    task_pr_scale = 1.
    verbose = False
    
    day_duration = 600
    params = {
        "time_zone": 0
    }

    scaling_inputs = {
        'scale_type': "no_scaling",
        'scale_min':  None,
        'scale_max':  None
    }
    
    timer = dict()

    assign_smdp_points(
        d_bm, timer=timer, choice_mode=choice_mode, day_duration=day_duration,
        gamma=gamma, json=False, scaling_inputs=scaling_inputs, verbose=verbose,
        goal_pr_loc=goal_pr_loc, goal_pr_scale=goal_pr_scale,
        task_pr_loc=task_pr_loc, task_pr_scale=task_pr_scale
    )
