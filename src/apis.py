from src.point_scalers import scale_optimal_rewards
from src.utils import tree_to_old_structure
from src.utils import incentivize_forced_pull
from src.schedulers.schedulers import schedule_tasks_for_today

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


def assign_smdp_points(projects, current_day, day_duration, leaf_projects,
                       smdp_params, timer, json=True, start_time=0, verbose=False):
    
    # # TODO: Remove (!)
    # for goal in leaf_projects:
    #     for task in goal["ch"]:
    #         print(task["nm"], task["scheduled_today"], task["today"])
    
    # TODO: Remove (!)
    print("Before tree parsing...")
    
    if json:

        """ Convert real goals from JSON to Goal class objects """
        tic = time.time()
        goals = tree_to_old_structure(projects, smdp_params)
        timer["Run SMDP - Converting JSON to objects"] = time.time() - tic

    else:
        goals = projects
        
    # TODO: Remove (!)
    # for goal in goals:
    #     goal.print_recursively(level=0, indent=2)
        
    # TODO: Remove (!)
    print("Generated class tree!")
    
    """ Add them together into a single list """
    tic = time.time()
    to_do_list = ToDoList(goals,
                          gamma=smdp_params["gamma"],
                          loss_rate=smdp_params["loss_rate"],
                          num_bins=smdp_params["num_bins"],
                          penalty_rate=smdp_params["penalty_rate"],
                          slack_reward_rate=smdp_params["slack_reward"],
                          start_time=start_time)
    timer["Run SMDP - Creating ToDoList object"] = time.time() - tic
    
    # TODO: Remove (!)
    print("Created ToDoList object!")

    """ Solve to-do list """
    tic = time.time()
    to_do_list.solve(available_time=day_duration, verbose=verbose)
    timer["Run SMDP - Solving SMDP"] = time.time() - tic
    
    # TODO: Remove (!)
    print("Solved SMDP!")
    
    # print("===== TO-DO LIST =====")
    # pprint(to_do_list.Q)
    # print()
    #
    # pprint(to_do_list.R)
    # print()

    # for goal in to_do_list.get_goals():
    #     print(f"===== {goal.get_description()} =====")
    #
    #     pprint(goal.Q)
    #     print()
    #
    #     pprint(goal.R)
    #     print()
    #
    #     print("<<<<< Q(s0) >>>>>")
    #     for task, q in goal.Q_s0.items():
    #         print(task.get_description(), q)
    #     print()

    """ Compute pseudo-rewards """
    tic = time.time()
    prs = compute_start_state_pseudo_rewards(to_do_list,
                                             bias=smdp_params["bias"],
                                             scale=smdp_params["scale"])
    timer["Run SMDP - Compute pseudo-rewards"] = time.time() - tic
    
    # TODO: Remove (!)
    print("Computed PRs!")
    
    # Update bias and scale parameters
    smdp_params["bias"] = prs["bias"]
    smdp_params["scale"] = prs["scale"]
    
    # Unpack pseudo-rewards | TODO: Remove (?)
    # optimal_tasks = prs["optimal_tasks"]
    # suboptimal_tasks = prs["suboptimal_tasks"]
    # slack_tasks = prs["slack_tasks"]
    
    incentivized_tasks = prs["incentivized_tasks"]
    id2pr = prs["id2pr"]  # TODO: Remove (!)
    
    # TODO: Potentially unnecessary (?!)
    # """ Run SMDP - Scaling rewards
    #     Scale task values according to the provided scaling function """
    # tic = time.time()
    #
    # if len(incentivized_tasks) > 0:
    #     scale_optimal_rewards(incentivized_tasks,
    #                           scale_min=smdp_params["scale_min"],
    #                           scale_max=smdp_params["scale_max"],
    #                           scale_type=smdp_params["scale_type"])
    #
    # timer["Run SMDP - Scaling rewards"] = time.time() - tic

    # TODO: Remove (!)
    print("Scaled rewards!")

    # Convert task queue to a list
    optimal_tasks = list(incentivized_tasks)
    
    # Sort task in decreasing order w.r.t. optimal reward
    optimal_tasks.sort(key=lambda task: -task.get_optimal_reward())
    
    if json:
    
        """ Run SMDP - Scheduling tasks """
        tic = time.time()

        today_tasks = schedule_tasks_for_today(leaf_projects, optimal_tasks,
                                               current_day=current_day,
                                               duration_remaining=day_duration)
    
        timer["Run SMDP - Scheduling tasks"] = time.time() - tic
        
        # """ Run SMDP - Incentivizing forced pull """
        # tic = time.time()
        #
        # # TODO: Potentially unnecessary (?!)
        # forced_pull = incentivize_forced_pull(leaf_projects, pr_dict=id2pr)
        # today_tasks.extend(forced_pull)
        #
        # timer["Run SMDP - Incentivizing forced pull"] = time.time() - tic

    else:
        today_tasks = optimal_tasks

    # TODO: Remove (!)
    print("Scheduled tasks!")

    # if json:
    #
    #     # Add slack tasks to output
    #     for task in slack_tasks:
    #
    #         goal_list = list(task.get_goals())
    #         goal = goal_list[0]
    #
    #         task_json = {
    #             'completed':         False,
    #             'daily':             False,
    #             'day_datetime':      None,
    #             'days':              [False, False, False, False, False, False, False],
    #             'deadline':          task.get_deadline(),
    #             'deadline_datetime': task.get_deadline_datetime(),
    #             'future':            False,
    #             'repetitive_days':   [False, False, False, False, False, False,
    #                                   False],
    #             'scheduled_today':   True,
    #
    #             'est':               25,
    #             'id':                f"sa{goal.get_idx()}",
    #             'lm':                0,
    #             'nm':                f"{goal.get_idx() + 1}) {task.get_description()}",
    #             'parentId':          goal.get_id(),
    #             'pcp':               False,
    #             'pph':               0,
    #             'today':             True,
    #             'val':               0
    #         }
    #
    #         today_tasks.append(task_json)
    #
    #     # Sort tasks in reversed order
    #     # today_tasks.reverse()
    
    print("Done!")
    
    return today_tasks
