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


def assign_smdp_points(projects, current_day, day_duration, smdp_params, timer,
                       json=True, start_time=0, verbose=False):
    if json:

        """ Convert real goals from JSON to Goal class objects """
        tic = time.time()
        goals = tree_to_old_structure(projects, smdp_params)
        timer["Run SMDP - Converting JSON to objects"] = time.time() - tic

    else:
        goals = projects
    
    """ Add them together into a single list """
    tic = time.time()
    to_do_list = ToDoList(goals,
                          gamma=smdp_params["gamma"],
                          slack_reward=smdp_params["slack_reward"],
                          start_time=start_time)
    timer["Run SMDP - Creating ToDoList object"] = time.time() - tic

    """ Solve to-do list """
    tic = time.time()
    to_do_list.solve(available_time=day_duration, verbose=verbose)
    timer["Run SMDP - Solving SMDP"] = time.time() - tic

    """ Compute pseudo-rewards """
    tic = time.time()
    prs = compute_start_state_pseudo_rewards(to_do_list,
                                             bias=smdp_params["bias"],
                                             scale=smdp_params["scale"])
    timer["Run SMDP - Compute pseudo-rewards"] = time.time() - tic
    
    # Unpack pseudo-rewards
    optimal_tasks = prs["optimal_tasks"]
    suboptimal_tasks = prs["suboptimal_tasks"]
    slack_tasks = prs["slack_tasks"]
    id2pr = prs["id2pr"]
    
    """ Run SMDP - Scaling rewards
        Scale task values according to the provided scaling function """
    tic = time.time()

    if len(optimal_tasks + suboptimal_tasks) > 0:
        scale_optimal_rewards(optimal_tasks + suboptimal_tasks,
                              scale_min=smdp_params["scale_min"],
                              scale_max=smdp_params["scale_max"],
                              scale_type=smdp_params["scale_type"])

    timer["Run SMDP - Scaling rewards"] = time.time() - tic
    
    # Convert task queue to a list
    optimal_tasks = list(optimal_tasks)
    
    # Sort task in decreasing order w.r.t. optimal reward
    optimal_tasks.sort(key=lambda task: -task.get_optimal_reward())
    
    if json:
    
        """ Run SMDP - Scheduling tasks """
        tic = time.time()
    
        today_tasks = schedule_tasks_for_today(projects, optimal_tasks,
                                               current_day=current_day,
                                               duration_remaining=day_duration)
    
        timer["Run SMDP - Scheduling tasks"] = time.time() - tic
        
        """ Run SMDP - Incentivizing forced pull """
        tic = time.time()

        forced_pull = incentivize_forced_pull(projects, pr_dict=id2pr)
        today_tasks.extend(forced_pull)
    
        timer["Run SMDP - Incentivizing forced pull"] = time.time() - tic

    else:
        today_tasks = optimal_tasks

    if json:
        
        # Add slack tasks to output
        for task in slack_tasks:
            goal = list(task.get_goals())[0]

            task_json = {
                'completed':         False,
                'daily':             False,
                'day_datetime':      None,
                'days':              [False, False, False, False, False, False, False],
                'deadline':          task.get_deadline(),
                'deadline_datetime': task.get_deadline_datetime(),
                'future':            False,
                'repetitive_days':   [False, False, False, False, False, False,
                                      False],
                'scheduled_today':   True,

                'est':               25,
                'id':                f"sa{goal.get_idx()}",
                'lm':                0,
                'nm':                f"{goal.get_idx() + 1}) {task.get_description()}",
                'parentId':          goal.get_id(),
                'pcp':               False,
                'pph':               0,
                'today':             True,
                'val':               0
            }

            today_tasks.append(task_json)
    
        # Sort tasks in reversed order
        # today_tasks.reverse()
    
    return today_tasks
