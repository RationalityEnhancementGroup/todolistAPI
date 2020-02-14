import cherrypy
import re

from copy import deepcopy
from datetime import datetime, timedelta
from math import ceil
from pprint import pprint
from string import digits

from todolistMDP.to_do_list import Goal, Task

deadline_regex = r"DUE:\s*([0-9][0-9][0-9][0-9][\-\.\\\/]+(0[1-9]|1[0-2]|[1-9])[\-\.\\\/]+([0-2][0-9]|3[0-1]|[1-9]))(\s+([0-1][0-9]|2[0-3]|[0-9])[\-\:\;\.\,]+([0-5][0-9]|[0-9])|)"
goal_code_regex = r"#CG(\d+|&|_|\^)"
time_est_regex = r"(?:^||>)\(?~~\s*\d+[\.\,]*\d*\s*(?:((h(?:our|r)?)|(m(?:in)?)))s?\)?(?:|[^\da-z.]|$)"
today_regex = r"#today(?:\b|)"
total_value_regex = r"(?:^||>)\(?==\s*(\d+)\)?(?:|\b|$)"

DEADLINE_YEAR_LIMIT = 2100


def are_there_tree_differences(old_tree, new_tree):
    """
    input: two trees
    output: boolean of whether or not we need to rerun the point calculations
    (e.g. we don't need to if only day durations change or #today has been added)
    """
    def create_tree_dict(tree):
        """
        input: parsed tree
        output: a dict with info we may want to use to compare trees
        """
        final_dict = {}
        for goal in tree:
            final_dict[goal["id"]] = (goal["deadline_datetime"], goal["value"])
            for task in goal["ch"]:
                final_dict[task["id"]] = \
                    (task["deadline_datetime"], task["est"])
        return final_dict

    if len(set(create_tree_dict(old_tree).items()) ^
           set(create_tree_dict(new_tree).items())) == 0:
        return False
    else:
        return True


def clean_output(task_list, round_param):
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
        task["val"] = round(task["val"], round_param)
    
    return task_list


def create_projects_to_save(projects):
    projects_to_save = deepcopy(projects)
    for project in projects_to_save:
        del project["nm"]
        try:
            del project["no"]
        except:
            pass
        for task in project["ch"]:
            del task["nm"]
            try:
                del task["no"]
            except:
                pass
    return projects_to_save


def flatten_intentions(projects):
    for goal in projects:
        for task in goal["ch"]:
            if "ch" in task:
                goal["ch"].extend(task["ch"])
                del task["ch"]
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
        if (misc_goal["deadline"]) is None:
            misc_goal['deadline'] = latest_deadline

        for task in misc_goal['ch']:
            task_goal = deepcopy(misc_goal)

            if task["deadline"]:
                task_goal["deadline"] = task["deadline"]

            if task["deadline_datetime"]:
                task_goal["deadline_datetime"] = task["deadline_datetime"]
                
            task_goal["est"] = task["est"]
            task_goal['id'] = task['id']
            task_goal['nm'] = task['nm']
            task_goal["parentId"] = task["parentId"]
            
            task_goal["value"] *= task["est"] / misc_goal["est"]
            task_goal["value"] = ceil(task_goal["value"])
            
            task_goal['ch'] = [task]

            misc_tasks += [task_goal]
    
    return misc_tasks


def parse_current_intentions_list(current_intentions):
    """
    Extracts necessary information from CompliceX's current intentions list.
    
    Args:
        current_intentions: List of current intentions on CompliceX.

    Returns:
        Dictionary of all parsed current intentions.
    """
    def get_wf_task_id(task_name):
        """
        Extracts the WorkFlowy ID from the name of the task.
        Args:
            task_name: Task name

        Returns:
            Task ID
        """
        return task_name.split("$wf:")[-1]

    # Dictionary of all parsed current intentions
    current_intentions_dict = dict()
    
    for task in current_intentions:
        task_dict = dict()
        
        # Get necessary information
        task_dict["id"] = get_wf_task_id(task["t"])
        task_dict["d"] = task["d"] if "d" in task.keys() else False
        task_dict["est"] = process_time_est(task["t"])
        task_dict["vd"] = task["vd"]
        
        # Add current task to the dictionary of all parsed current intentions
        current_intentions_dict[task_dict["id"]] = task_dict
        
    return current_intentions_dict


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
    
    
def parse_tree(projects, current_intentions, allowed_task_time, today_minutes,
               typical_minutes, default_duration, default_deadline):
    """
    This function reads in a flattened project tree and parses fields like goal
    code, total value, duration and deadline
    """
    def get_wf_task_id(task_name):
        return task_name.split("-")[-1]
    
    real_goals = []
    misc_goals = []

    for goal in projects:
        # Initialize goal time estimation
        goal["est"] = 0

        # Extract goal information
        goal["code"] = re.search(goal_code_regex, goal["nm"], re.IGNORECASE)[1]
        goal_deadline = re.search(deadline_regex, goal["nm"], re.IGNORECASE)
        
        # If no deadline has been provided --> Misc goal
        if goal["code"][0] not in digits+"^":
            try:
                goal["deadline"], goal["deadline_datetime"] = \
                    process_deadline(goal_deadline, today_minutes,
                                     typical_minutes, default_deadline)
            except Exception as error:
                raise Exception(f"Goal {goal['nm']}: {str(error)}")
            misc_goals += [goal]
        else:
            # Process goal deadline and check whether the value is valid
            try:
                goal["deadline"], goal["deadline_datetime"] = \
                    process_deadline(goal_deadline, today_minutes,
                                     typical_minutes, default_deadline)
            except Exception as error:
                raise Exception(f"Goal {goal['nm']}: {str(error)}")

            real_goals += [goal]
            
        for task in goal["ch"]:
            
            # Get the last part of the HEX ID code for the task in WorkFlowy
            task_id = get_wf_task_id(task["id"])
            
            # Get task deadline (if provided)
            task_deadline = re.search(deadline_regex, task["nm"], re.IGNORECASE)
            
            if task_deadline:
                try:
                    task["deadline"], task["deadline_datetime"] = \
                        process_deadline(task_deadline, today_minutes,
                                         typical_minutes)
                except Exception as error:
                    raise Exception(f"Task {task['nm']}: {str(error)}")

                # Check whether task deadline is after goal deadline
                if task["deadline"] > goal["deadline"]:
                    raise Exception(f"Task {task['nm']}: Task deadline should "
                                    f"be before goal's deadline.")
            else:
                task["deadline"] = None
                task["deadline_datetime"] = None
                
            # Check whether the task has already been scheduled in CompliceX or
            # completed in WorkFlowy
            if task_id in current_intentions.keys() or \
                    ("cp" in task.keys() and task["cp"] >= task["lm"]):
                task["completed"] = True
            else:
                task["completed"] = False
    
            # Process time estimation for a task
            try:
                task["est"] = \
                    process_time_est(task["nm"], allowed_task_time, default_duration)
            except Exception as error:
                raise Exception(f"Task {task['nm']}: {str(error)}")
            
            # Update goal time estimation
            goal["est"] += task["est"]
            
            # Check whether a task has been marked to be completed today
            task["today"] = process_today_code(task)
    
            task["parentId"] = goal["id"]
            task["pcp"] = False  # TODO: Not sure what this field is...
            
            # Append goal's name to task's name
            task["nm"] = goal["code"] + ") " + task["nm"]
            
        # Process goal value and check whether the value is valid
        try:
            goal["value"] = process_goal_value(goal)
        except Exception as error:
            raise Exception(f"Goal {goal['nm']}: {str(error)}")
    
    return real_goals, misc_goals


def process_deadline(deadline, today_minutes, typical_minutes,
                     default_deadline=None):
    # Time from which the deadlines are computed
    current_time = datetime.utcnow()
    if deadline is None:
        if default_deadline is not None:
            default_deadline_datetime = \
                timedelta(days=int(default_deadline))
            deadline = \
                re.search(deadline_regex, "DUE:" +
                          (current_time + default_deadline_datetime).strftime("%Y-%m-%d"),
                          re.IGNORECASE)
        else:
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
        
        if int(year) >= DEADLINE_YEAR_LIMIT:
            raise Exception(f"Deadline too far in the future!")
    
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
    deadline_datetime = datetime.strptime(deadline, "%Y-%m-%d %H:%M")
    td = deadline_datetime - current_time
    
    # Convert difference between deadlines into minutes
    # (ignoring remaining seconds)
    deadline_value = today_minutes + \
                     ((td.days - 1) * typical_minutes) + (td.seconds // 60)

    # Check whether it is in the future
    if deadline_value <= 0:
        raise Exception(f"Deadline not in the future!")

    return deadline_value, deadline_datetime


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


def process_time_est(task_name, allowed_task_time=float('inf'), default_duration=None):
    try:
        time_est = re.search(time_est_regex, task_name, re.IGNORECASE)[0]
    except:
        if default_duration is not None:
            time_est = "~~"+str(default_duration)+"min"
        else:
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
    
    # Check whether the value is valid
    if duration <= 0:
        raise Exception(f"{task_name}: Time estimation is not a "
                        f"positive number!")
    if duration > allowed_task_time:
        raise Exception(f"{task_name}: Time duration not allowed!")
    
    # Convert time to minutes. If fractional, get the higher rounded value!
    duration = int(ceil(duration))
    
    return duration


def process_today_code(task):
    today_code = re.search(today_regex, task["nm"], re.IGNORECASE)
    
    if today_code:  # ... is not None
        return True
    else:
        return False


def separate_tasks_with_deadlines(goals):
    tasks_with_deadlines = []
    
    for goal in goals:
        separated_tasks = []
        
        for task in goal["ch"]:
            if task["deadline"]:
                task_goal = deepcopy(goal)
                
                task_goal["deadline"] = task["deadline"]
                task_goal["deadline_datetime"] = task["deadline_datetime"]
                task_goal["est"] = task["est"]
                task_goal["id"] = task["id"]
                task_goal["nm"] = task["nm"]
                task_goal["parentId"] = task["parentId"]
                
                task_goal["value"] *= task["est"] / goal["est"]
                task_goal["value"] = ceil(task_goal["value"])

                task_goal["ch"] = [task]
    
                separated_tasks += [task_goal]
                
        tasks_with_deadlines += separated_tasks
        
        # Separate task from goal tasks & subtract time estimation and value
        for task in separated_tasks:
            goal["ch"].remove(task["ch"][0])
            goal["est"] = max(goal["est"] - task["est"], 0)
            goal["value"] = max(goal["value"] - task["value"], 0)

    return goals + tasks_with_deadlines
    

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
        for task in goal["ch"]:
            if task["deadline"] is None:
                task["deadline"] = goal["deadline"]
            task_list.append(task)
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

            # Create new task and add it to the task list
            tasks.append(Task(completed=task["completed"],
                              description=task["nm"],
                              task_id=task["id"],
                              time_est=task["est"],
                              prob=1))  # TODO: Probability of success

        # Create new goal and add it to the goal list
        goals.append(
            Goal(description=goal["nm"],
                 goal_id=goal["id"],
                 tasks=tasks,
                 rewards={goal["deadline"]: goal["value"]},
                 penalty=0))  # TODO: Penalty for missing a deadline
        
    return goals
