from src.point_scalers import scale_optimal_rewards
from src.utils import tree_to_old_structure
from src.schedulers.schedulers import schedule_items_for_today

from todolistMDP.smdp_utils import *
from todolistMDP.to_do_list import *


def assign_constant_points(projects, default_task_value=10):
    """
    Takes in parsed project tree, with one level of tasks
    Outputs project tree with constant points assigned
    """
    for goal in projects:
        for task in goal["ch"]:
            task["val"] = default_task_value
    return projects


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


def assign_smdp_points(projects, all_json_items, current_day, day_duration,
                       smdp_params, timer, start_time=0,
                       verbose=False):
    
    """ Convert tree from JSON to Item class objects """
    tic = time.time()
    goals = tree_to_old_structure(projects, smdp_params)
    timer["Run SMDP - Converting JSON to objects"] = time.time() - tic

    """ Add them together into a single list """
    tic = time.time()
    # to_do_list = ToDoList(goals,
    #                       gamma=smdp_params["gamma"],
    #                       loss_rate=smdp_params["loss_rate"],
    #                       num_bins=smdp_params["num_bins"],
    #                       penalty_rate=smdp_params["penalty_rate"],
    #                       slack_reward_rate=smdp_params["slack_reward"],
    #                       start_time=start_time)
    to_do_list = MainToDoList(goals,
                              gamma=smdp_params["gamma"],
                              loss_rate=smdp_params["loss_rate"],
                              num_bins=smdp_params["num_bins"],
                              penalty_rate=smdp_params["penalty_rate"],
                              slack_reward_rate=smdp_params["slack_reward"],
                              start_time=start_time)
    timer["Run SMDP - Creating ToDoList object"] = time.time() - tic

    # If not task-level MDP, extend "day" duration
    if smdp_params["sub_goal_max_time"] != 0:
        if "time_frame" in smdp_params.keys():
            day_duration = smdp_params["time_frame"]
        else:
            day_duration = smdp_params["sub_goal_max_time"]

    """ Parse sub-goals """
    tic = time.time()
    for goal in goals:
        
        # Parse sub-goals
        sub_goals = goal.parse_sub_goals(
            min_time=smdp_params["sub_goal_min_time"],
            max_time=smdp_params["sub_goal_max_time"]
        )
        
        # Set sub-goals
        goal.add_items(sub_goals, available_time=deepcopy(day_duration),
                       prepare_solve=True)
        
    timer["Run SMDP - Parsing sub-goals"] = time.time() - tic

    """ Solve to-do list """
    tic = time.time()
    # to_do_list.solve(in_depth=False, verbose=verbose)
    solved_dict = to_do_list.solve()
    timer["Run SMDP - Solving SMDP"] = time.time() - tic
    
    # """ Compute pseudo-rewards """
    # tic = time.time()
    # prs = compute_start_state_pseudo_rewards(to_do_list,
    #                                          bias=smdp_params["bias"],
    #                                          scale=smdp_params["scale"])
    # timer["Run SMDP - Compute pseudo-rewards"] = time.time() - tic
    #
    # # Update bias and scale parameters
    # smdp_params["bias"] = prs["bias"]
    # smdp_params["scale"] = prs["scale"]
    #
    # # Get incentivized tasks
    # incentivized_tasks = prs["incentivized_items"]
    #
    # """ Run SMDP - Scaling rewards
    #     Scale task values according to the provided scaling function """
    # tic = time.time()
    #
    # if smdp_params["scale_type"] is not None and len(incentivized_tasks) > 0:
    #     scale_optimal_rewards(incentivized_tasks,
    #                           scale_min=smdp_params["scale_min"],
    #                           scale_max=smdp_params["scale_max"],
    #                           scale_type=smdp_params["scale_type"])
    #
    # timer["Run SMDP - Scaling rewards"] = time.time() - tic

    # From dict get tasks
    optimal_tasks_names = {}
    optimal_tasks = {}
    for i in range(to_do_list.num_tasks):
        optimal_tasks_names[to_do_list.tasks[i].description] = solved_dict[to_do_list.tasks[i].description]
        optimal_tasks[to_do_list.tasks[i].description] = to_do_list.tasks[i]

    # Sort task in decreasing order w.r.t. optimal reward
    sorted_optimal_tasks_names = dict(sorted(optimal_tasks_names.items(), key=lambda item: -item[1]))
    optimal_tasks_list = []
    for task in sorted_optimal_tasks_names:
        optimal_tasks_list.append(optimal_tasks[task])

    
    """ Run SMDP - Scheduling items """
    tic = time.time()

    today_tasks = schedule_items_for_today(all_json_items, optimal_tasks_list,
                                           current_day=current_day,
                                           duration_remaining=day_duration)

    timer["Run SMDP - Scheduling items"] = time.time() - tic
    
    return today_tasks
