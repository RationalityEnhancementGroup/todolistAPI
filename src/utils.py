import cherrypy
import re

from copy import deepcopy
from datetime import datetime
from todolistMDP.to_do_list import Goal, Task

goalCodeRegex = r"#CG(\d+|&|_)"
totalValueRegex = r"(?:^| |>)\(?==(\d+)\)?(?:\b|$)"
timeEstimateRegex = r"(?:^| |>)\(?~~(\d+|\.\d+|\d+.\d+)(?:(h(?:ou)?(?:r)?)?(m(?:in)?)?)?s?\)?([^\da-z.]|$)"
deadlineRegex = r"DUE:(\d\d\d\d-\d+-\d+)(\s+\d\:\d\d|\s+\d\d\:\d\d|\s*$)"
todayRegex = r"#today(?:\b|$)"


def are_there_tree_differences(old_tree, new_tree):
    """
    input: two trees
    output: boolean of whether or not we need to rerun the point calculations
    (e.g. we don't need to if only day durations change or #today has been added)
    """
    if len(set(create_tree_dict(old_tree).items()) ^
           set(create_tree_dict(new_tree).items())) == 0:
        return False
    else:
        return True


def clean_output(task_list):
    """
    Input is list of tasks
    Outputs list of tasks for today with fields:
        id, nm, lm, parentId, pcp, est, val (=reward)
    """
    keys_needed = ["id", "nm", "lm", "parentId", "pcp", "est", "val"]
    
    # for now only look at first dictionary
    current_keys = set(task_list[0].keys())
    extra_keys = list(current_keys - set(keys_needed))
    missing_keys = list(set(keys_needed) - current_keys)
    
    for extra_key in extra_keys:
        for task in task_list:
            if extra_key in task:
                del task[extra_key]
    
    for missing_key in missing_keys:
        for task in task_list:
            if missing_key not in task:
                task[missing_key] = None
    
    for task in task_list:
        task["val"] = round(task["val"])
    
    return task_list


def create_tree_dict(tree):
    """
    input: parsed tree
    output: a dict with info we may want to use to compare trees
    # TODO probably a better way to do this in are_there_tree_differences
    """
    final_dict = {}
    for goal in tree:
        final_dict[goal["id"]] = (goal["deadline"], goal["value"])
        for task in goal["ch"]:
            final_dict[task["id"]] = task["est"]
    return final_dict


def flatten_intentions(projects):
    for goal in projects:
        for child in goal["ch"]:
            if "ch" in child:
                goal["ch"].extend(child["ch"])
                del child["ch"]
    return projects


def misc_tasks_to_goals(real_goals, misc_goals, extra_time=0, small_value=1):
    real_goals.sort()
    latest_deadline = real_goals[-1].get_deadline_time()
    
    # Update latest deadline
    for misc_goal in misc_goals:
        for misc_task in misc_goal["ch"]:
            latest_deadline += misc_task["est"]
    
    latest_deadline += extra_time
    
    # Decompose misc goals into goals for each task of the goals
    misc_tasks = []
    for misc_goal in misc_goals:
        
        # Assign deadline and value for misc goal
        misc_goal['deadline'] = latest_deadline
        misc_goal['value'] = small_value
        
        for child in misc_goal['ch']:
            task_goal = deepcopy(misc_goal)
            
            task_goal['id'] = child['id']
            task_goal['nm'] = child['nm']
            
            task_goal['ch'] = [child]

            misc_tasks += [task_goal]
    
    return misc_tasks


def parse_hours(time_string, default_hours=8):
    try:
        return int(re.search(totalValueRegex, time_string, re.IGNORECASE)[1])
    except:
        return default_hours
    
    
def parse_tree(projects, allowed_task_time, typical_hours):
    """
    This function reads in a flattened project tree and parses fields like goal
    code, total value, duration and deadline
    """
    real_goals = []
    misc_goals = []
    
    for goal in projects:
        # Extract goal information
        goal["code"] = re.search(goalCodeRegex, goal["nm"], re.IGNORECASE)[1]
        goal_value = re.search(totalValueRegex, goal["nm"], re.IGNORECASE)
        goal_deadline = re.search(deadlineRegex, goal["nm"], re.IGNORECASE)
        
        goal["est"] = 0
        for task in goal["ch"]:
            task["est"] = process_time_est(task, allowed_task_time)
    
            # Check whether a task has been marked to be completed today
            task["today"] = process_today_code(task)
    
            task["parentId"] = goal["id"]
            task["pcp"] = False  # TODO: Not sure what this field is...
            
            goal["est"] += task["est"]
            
        goal["deadline"] = None
        
        # Process goal value and check whether the value is valid
        try:
            goal["value"] = process_goal_value(goal_value)
        except Exception as error:
            raise Exception(f"{goal['nm']}: {str(error)}")
            # raise Exception(f"Goal \"{goal['nm']}\" has "
            #                 f"invalid goal value!")

        # If no deadline has been provided --> Misc goal
        if goal_deadline is None:
            misc_goals += [goal]
        else:
            # Process goal deadline and check whether the value is valid
            try:
                goal["deadline"] = process_goal_deadline(goal_deadline,
                                                         typical_hours)
            except Exception as error:
                raise Exception(f"{goal['nm']}: {str(error)}")
                # raise Exception(f"Goal \"{goal['nm']}\" has "
                #                 f"invalid goal deadline!")

            real_goals += [goal]
    
    return real_goals, misc_goals


def process_goal_deadline(goal_deadline, typical_hours):
    # TODO: Date at the moment... Enter some delay?
    current_date = datetime.now()  # .date()
    goal_deadline = goal_deadline[1].strip()
    
    # If no time is included
    if not (' ' in goal_deadline and ':' in goal_deadline):
        goal_deadline += ' 23:59'  # End of the day
    
    goal_deadline = datetime.strptime(goal_deadline, "%Y-%m-%d %H:%M")
    td = goal_deadline - current_date
    
    # Convert difference between deadlines into minutes
    # (ignoring remaining seconds)
    goal_deadline = (td.days * typical_hours * 60) + (td.seconds // 60)

    try:
        goal_deadline = int(goal_deadline)
    except:
        raise Exception("Invalid goal deadline!")
    
    # Check whether it is in the future
    if goal_deadline <= 0:
        raise Exception("Goal deadline not in the future!")

    return goal_deadline


def process_goal_value(goal_value):
    goal_value = int(goal_value[1])

    if goal_value <= 0:
        raise Exception("Goal value not a positive number!")

    return goal_value


def process_time_est(task, allowed_task_time):
    time_est = re.search(timeEstimateRegex, task["nm"], re.IGNORECASE)
    
    if time_est is None:
        raise Exception("Missing task time estimation!")
    
    duration = int(time_est[1])
    if time_est[2] is not None:  # Hours --> Convert to minutes
        duration *= 60
    
    if duration > allowed_task_time:
        raise Exception(f"Task \"{task['nm']}\" has duration more than the "
                        f"allowed time duration!")
    
    return duration


def process_today_code(task):
    today_code = re.search(todayRegex, task["nm"], re.IGNORECASE)
    if today_code:  # ... is not None
        return True
    else:
        return False


def task_dict_from_projects(projects):
    return {
        task["id"]: task
        for goal in projects
        for task in goal["ch"]
    }


def task_list_from_projects(projects):
    task_list = []
    for goal in projects:
        task_list.extend(goal["ch"])
    return task_list


def tree_to_old_structure(projects):
    """
    input: parsed tree
    output: structure that can be inputted to old project code
    """
    goals = []
    for goal in projects:
        
        # Get list of tasks
        tasks = []
        for task in goal['ch']:
            
            # Get time estimation and check whether the value is valid
            # --> Upper limit done in the parse_tree function
            if task['est'] <= 0:
                raise Exception("Time estimation not a positive number!")

            # TODO: Probability of success

            tasks.append(Task(description=task['nm'],
                              task_id=task['id'],
                              time_est=task['est'],
                              prob=1))

        # TODO: Penalties
        
        # Create new goal and add it to the goal list
        goals.append(
            Goal(description=goal["nm"],
                 goal_id=goal["id"],
                 tasks=tasks,
                 reward={goal["deadline"]: goal["value"]},
                 penalty=0))
        
    return goals
