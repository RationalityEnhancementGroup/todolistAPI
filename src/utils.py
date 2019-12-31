import cherrypy
import re

from copy import deepcopy
from datetime import datetime
from math import ceil
from string import digits
from todolistMDP.to_do_list import Goal, Task

deadline_regex = r"DUE:\s*(20[1-9][0-9][\-\.\\\/]+(0[1-9]|1[0-2]|[1-9])[\-\.\\\/]+([0-2][0-9]|3[0-1]|[1-9]))(\s+([0-1][0-9]|2[0-3]|[0-9])[\-\:\;\.\,]+([0-5][0-9]|[0-9])|)"
goal_code_regex = r"#CG(\d+|&|_)"
time_est_regex = r"(?:^||>)\(?~~\s*\d+[\.\,]*\d*\s*(?:((h(?:our|r)?)|(m(?:in)?)))s?\)?(?:|[^\da-z.]|$)"
today_regex = r"#today(?:\b|)"
total_value_regex = r"(?:^||>)\(?==\s*(\d+)\)?(?:|\b|$)"


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


def create_projects_to_save(projects):
    projects_to_save = deepcopy(projects)
    for project in projects_to_save:
        del project["nm"]
        for task in project["ch"]:
            del task["nm"]
    return projects_to_save


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


def misc_tasks_to_goals(real_goals, misc_goals, extra_time=0):
    real_goals.sort()
    latest_deadline = real_goals[-1].get_latest_deadline_time()
    
    # Update latest deadline
    total_misc_time_est = 0

    for misc_goal in misc_goals:
        misc_goal["est"] = 0
        
        for misc_task in misc_goal["ch"]:
            misc_goal["est"] += misc_task["est"]
            
        total_misc_time_est += misc_goal["est"]
    
    latest_deadline += total_misc_time_est + extra_time
    
    # Decompose misc goals into goals for each task of the goals
    misc_tasks = []
    for misc_goal in misc_goals:
        
        # Assign deadline and value for misc goal
        misc_goal['deadline'] = latest_deadline

        for child in misc_goal['ch']:
            task_goal = deepcopy(misc_goal)
            
            task_goal['id'] = child['id']
            task_goal['nm'] = child['nm']
            
            task_goal["value"] *= child["est"] / misc_goal["est"]
            task_goal["value"] = ceil(task_goal["value"])
            
            task_goal['ch'] = [child]

            misc_tasks += [task_goal]
    
    return misc_tasks


def parse_error_info(error):
    """
    Removes personal info and returns the exception info.

    Args:
        error: Error message as string

    Returns:
        Exception info without personal data.
    """
    return error.split(": ")[-1]


def parse_hours(time_string):
    return int(re.search(total_value_regex, time_string, re.IGNORECASE)[1])
    
    
def parse_tree(projects, allowed_task_time, typical_hours):
    """
    This function reads in a flattened project tree and parses fields like goal
    code, total value, duration and deadline
    """
    real_goals = []
    misc_goals = []
    
    for goal in projects:
        # Extract goal information
        goal["code"] = re.search(goal_code_regex, goal["nm"], re.IGNORECASE)[1]
        goal_deadline = re.search(deadline_regex, goal["nm"], re.IGNORECASE)
        
        goal["est"] = 0
        for task in goal["ch"]:
            try:
                task["est"] = process_time_est(task, allowed_task_time)
            except Exception as error:
                raise Exception(f"Task {task['nm']}: {str(error)}")

            # Check whether a task has been marked to be completed today
            task["today"] = process_today_code(task)
    
            task["parentId"] = goal["id"]
            task["pcp"] = False  # TODO: Not sure what this field is...
            
            # Add task time estimation to total goal time estimation
            goal["est"] += task["est"]
            
            # Append goal's name to task's name
            task["nm"] = goal["code"] + ") " + task["nm"]
            
        # Process goal value and check whether the value is valid
        try:
            goal["value"] = process_goal_value(goal)
        except Exception as error:
            raise Exception(f"Goal {goal['nm']}: {str(error)}")

        # If no deadline has been provided --> Misc goal
        if goal["code"][0] not in digits:
            goal["deadline"] = None
            misc_goals += [goal]
        else:
            # Process goal deadline and check whether the value is valid
            try:
                goal["deadline"] = process_goal_deadline(goal_deadline,
                                                         typical_hours)
            except Exception as error:
                raise Exception(f"Goal {goal['nm']}: {str(error)}")

            real_goals += [goal]
    
    return real_goals, misc_goals


def process_goal_deadline(deadline, typical_hours):
    # TODO: Date at the moment... Enter some delay?
    current_date = datetime.now()  # .date()

    if deadline is None:
        raise Exception("Invalid or no deadline provided!")

    # Remove empty spaces at the beginning and the end of the string
    deadline = deadline[0].strip()
    
    # Remove "DUE:\s*"
    deadline = re.sub(r"DUE:\s*", "", deadline, re.IGNORECASE)
    
    # Split date and time
    deadline_args = re.split(r"\s+", deadline)
    
    if len(deadline_args) >= 1:
        # Parse date
        try:
            year, month, day = re.split(r"[\-\.\\\/]+", deadline_args[0])
        except:
            raise Exception(f"Invalid deadline date!")
    
        if len(deadline_args) == 2:
            # Parse time
            try:
                hours, minutes = re.split(r"[\-\:\;\.\,]+", deadline_args[1])
            except:
                raise Exception(f"Invalid deadline time!")
            
        else:
            hours, minutes = '23', '59'  # End of the day
    
        deadline = f"{year}-{month}-{day} {hours}:{minutes}"
        
    # Convert deadline into datetime object
    deadline_value = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
    td = deadline_value - current_date
    
    # Convert difference between deadlines into minutes
    # (ignoring remaining seconds)
    deadline_value = (td.days * typical_hours * 60) + (td.seconds // 60)

    # Check whether it is in the future
    if deadline_value <= 0:
        raise Exception(f"Deadline not in the future!")

    return deadline_value


def process_goal_value(goal):
    goal_value = re.search(total_value_regex, goal["nm"], re.IGNORECASE)
    
    if goal_value is None:
        raise Exception("No value provided!")
    
    # Parse value
    goal_value = int(goal_value[1])

    # Check whether it is a positive number
    if goal_value <= 0:
        raise Exception("Value not a positive number!")

    return goal_value


def process_time_est(task, allowed_task_time):
    try:
        time_est = re.search(time_est_regex, task["nm"], re.IGNORECASE)[0]
    except:
        raise Exception("No time estimation or invalid time estimation provided!")

    # Get time units (the number of hours or minutes) | Allows time fractions
    try:
        duration = re.search(r"\d+[\.\,]*\d*", time_est, re.IGNORECASE)[0]
        duration = re.split(r"[\.\,]+", duration)
        duration = ".".join(duration)
        duration = float(duration)
    except:
        raise Exception("Invalid time estimate!")

    # Get unit measurement info
    in_hours = re.search(r"h(?:our|r)?s?", time_est, re.IGNORECASE)
    # in_minutes = re.search(r"m(?:in)?s?", time_est, re.IGNORECASE)
    
    # If in hours --> Convert to minutes
    if in_hours:
        duration *= 60
    
    if duration > allowed_task_time:
        raise Exception(f"Time duration not allowed!")
    
    # Convert time to minutes. If fractional, get the higher rounded value!
    duration = int(ceil(duration))
    
    return duration


def process_today_code(task):
    today_code = re.search(today_regex, task["nm"], re.IGNORECASE)
    
    if today_code:  # ... is not None
        return True
    else:
        return False


def store_log(db_collection, log_dict, **params):
    """
    Stores the provided log dictionary in the DB collection with the additional
    (provided) parameters.
    
    Args:
        db_collection:
        log_dict:
        **params: Parameters to be stored, but NOT saved in the provided dict.
                  If you want to store the changes, then "catch" the returned
                  object after calling this function.

    Returns:
        Log dictionary with the existing + new parameters!
    """
    # Avoid overlaps
    log_dict = dict(log_dict)
    
    # Store additional info in the log dictionary
    for key in params.keys():
        log_dict[key] = params[key]

    log_dict["duration"] = str(datetime.now() - log_dict["start_time"])
    log_dict["timestamp"] = datetime.now()
    
    db_collection.insert_one(log_dict)  # Store info in DB collection
    
    return log_dict


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
                raise Exception(f"{task['nm']}: Time estimation is not a "
                                f"positive number!")

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
                 rewards={goal["deadline"]: goal["value"]},
                 penalty=0))
        
    return goals
