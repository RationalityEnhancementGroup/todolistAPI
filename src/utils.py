import cherrypy
import re

from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta
from math import ceil
from string import digits

from todolistMDP.to_do_list import Goal, Task

DATE_REGEX = r"([0-9][0-9][0-9][0-9][\-\.\\\/]+(0[1-9]|1[0-2]|[1-9])[\-\.\\\/]+([0-2][0-9]|3[0-1]|[1-9]))(\s+([0-1][0-9]|2[0-3]|[0-9])[\-\:\;\.\,]+([0-5][0-9]|[0-9])|)"
DEADLINE_REGEX = fr"DUE:\s*{DATE_REGEX}"
GOAL_CODES = r"(\d+|&|_|\^)"
HOURS_REGEX = r"(?:^||>)\(?\s*\d+[\.\,]*\d*\s*(?:hour)s?\)?(?:|[^\da-z.]|$)"
HTML_REGEX = r"<(\/|)(b|i|u)>"
INPUT_GOAL_CODE_REGEX = fr"#CG{GOAL_CODES}"
MINUTES_REGEX = r"(?:^||>)\(?\s*\d+[\.\,]*\d*\s*(?:minute)s?\)?(?:|[^\da-z.]|$)"
OUTPUT_GOAL_CODE_REGEX = fr"{GOAL_CODES}\)"
TIME_EST_REGEX = r"(?:^||>)\(?~~\s*\d+[\.\,]*\d*\s*(?:((h(?:our|r)?)|(m(?:in)?)))s?\)?(?:|[^\da-z.]|$)"
TOTAL_VALUE_REGEX = r"(?:^||>)\(?==\s*((-|)\d+)\)?(?:|\b|$)"

DEADLINE_YEAR_LIMIT = 2100
LARGE_NUMBER = 1000000
WEEKDAYS = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}
TAGS = ["future", "daily", "today", "weekdays", "weekends"] + \
       [weekday.lower() + r"(s)" for weekday in WEEKDAYS.values()] + \
       [weekday.lower() for weekday in WEEKDAYS.values()]


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
                    (task["day_datetime"], task["deadline_datetime"],
                     task["est"], task["task_days"])
        return final_dict

    if len(set(create_tree_dict(old_tree).items()) ^
           set(create_tree_dict(new_tree).items())) == 0:
        return False
    else:
        return True


def calculate_repetitive_tasks_time_est(projects, allowed_task_time,
                                       default_time_est):
    # Initialize total daily tasks time estimation for each weekday
    weekday_tasks_time_est = [0 for _ in range(7)]
    
    for goal in projects:
    
        # Remove formatting / HTML formatting
        goal["nm"] = re.sub(HTML_REGEX, "", goal["nm"],
                            count=LARGE_NUMBER, flags=re.IGNORECASE)
    
        # Initialize goal time estimation
        goal["est"] = 0
        
        for task in goal["ch"]:
    
            # Remove formatting / HTML tags
            task["nm"] = re.sub(HTML_REGEX, "", task["nm"],
                                count=LARGE_NUMBER, flags=re.IGNORECASE)

            # Process time estimation for a task
            try:
                task["est"] = \
                    process_time_est(task["nm"], allowed_task_time,
                                     default_time_est)
            except Exception as error:
                raise Exception(f"Task {task['nm']}: {str(error)}")
            
            # Update goal time estimation
            goal["est"] += task["est"]

            # Check whether weekday preferences are given
            task["task_days"], task["repetitive_task_days"] = \
                process_task_days(task)

            # Check whether it is a daily task
            task["daily"] = process_tagged_item("daily", task)
            
            if task["daily"]:
                for weekday in range(len(task["repetitive_task_days"])):
                    task["repetitive_task_days"][weekday] = True

            # Subtract time
            for weekday, repetitive in enumerate(task["repetitive_task_days"]):
                if repetitive:
                    weekday_tasks_time_est[weekday] -= task["est"]
            
    return weekday_tasks_time_est


def compute_latest_start_time(goal):
    goal_deadlines = goal["task_deadlines"]
    deadlines = sorted(list(goal_deadlines.keys()))
    
    # Initialize current time
    current_time = int(deadlines[0])
    
    for current_deadline in reversed(deadlines):
        current_time = min(current_time, int(current_deadline))
        current_time -= int(goal_deadlines[current_deadline])
    
        if current_time < 0:
            raise Exception(f'Goal {goal["nm"]} is unattainable!')
    
    return current_time


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


def date_str_to_datetime(date):
    # Remove empty spaces at the beginning and the end of the string
    date = date[0].strip()
    
    # Remove "DUE:\s*"
    date = re.sub(r"DUE:\s*", "", date, re.IGNORECASE)
    
    # Remove "#\s*"
    date = re.sub(r"#\s*", "", date, re.IGNORECASE)
    
    # Split date and time
    date_args = re.split(r"\s+", date)
    
    if len(date_args) >= 1:
        # Parse date
        try:
            year, month, day = re.split(r"[\-\.\\\/]+", date_args[0])
        except:
            raise Exception(f"Invalid deadline date!")
        
        if int(year) >= DEADLINE_YEAR_LIMIT:
            raise Exception(f"Deadline too far in the future!")
        
        if len(date_args) == 2:
            # Parse time
            try:
                hours, minutes = re.split(r"[\-\:\;\.\,]+", date_args[1])
            except:
                raise Exception(f"Invalid deadline time!")
        
        else:
            hours, minutes = '23', '59'  # End of the day
        
        date = f"{year}-{month}-{day} {hours}:{minutes}"
    
    # Convert deadline into datetime object
    date_datetime = datetime.strptime(date, "%Y-%m-%d %H:%M")
    
    return date_datetime


def flatten_intentions(projects):
    for goal in projects:
        for task in goal["ch"]:
            if "ch" in task:
                goal["ch"].extend(task["ch"])
                del task["ch"]
    return projects


def get_final_output(task_list, round_param, points_per_hour):
    """
    Input is list of tasks
    Outputs list of tasks for today with fields:
        id, nm, lm, parentId, pcp, est, val (=reward)
    """
    
    def get_human_readable_name(task):
        task_name = task["nm"]
        
        # Remove #date regex
        task_name = re.sub(fr"#\s*{DATE_REGEX}", "", task_name, re.IGNORECASE)

        # Remove deadline
        task_name = re.sub(DEADLINE_REGEX, "", task_name, re.IGNORECASE)
        
        # Remove time estimation
        task_name = re.sub(TIME_EST_REGEX, "", task_name, re.IGNORECASE)
        
        # Remove tags
        for tag in TAGS:
            tag_regex = get_tag_regex(tag)
            task_name = re.sub(tag_regex, "", task_name, re.IGNORECASE)
        
        task_name = task_name.strip()
        
        if len(re.sub(OUTPUT_GOAL_CODE_REGEX, "", task_name).strip()) == 0:
            raise NameError(f"Task {task['nm']} has no name!")
        
        # Append time information
        hours, minutes = task["est"] // 60, task["est"] % 60
        
        task_name += " (takes about "
        if hours > 0:
            if hours == 1:
                task_name += f"1 hour"
            else:
                task_name += f"{hours} hours"
        if minutes > 0:
            if hours > 0:
                task_name += " and "
            if minutes == 1:
                task_name += f"1 minute"
            else:
                task_name += f"{minutes} minutes"
        
        if task["deadline_datetime"] is not None:
            task_name += ", due on "

            td = task["deadline_datetime"] - datetime.utcnow()
            if td.days < 7:
                weekday = task["deadline_datetime"].weekday()
                task_name += WEEKDAYS[weekday]
            else:
                task_name += str(task["deadline_datetime"])[:-3]
        
        task_name += ")"
        
        return task_name
    
    keys_needed = ["id", "nm", "lm", "parentId", "pcp", "est", "val"]
    
    # for now only look at first dictionary
    current_keys = set(task_list[0].keys())
    extra_keys = list(current_keys - set(keys_needed))
    missing_keys = list(set(keys_needed) - current_keys)
    
    for task in task_list:
        task["nm"] = get_human_readable_name(task)
        
        if points_per_hour:
            task["val"] = str(round(task["pph"], round_param)) + '/h'
        else:
            task["val"] = round(task["val"], round_param)

        for extra_key in extra_keys:
            if extra_key in task:
                del task[extra_key]
        
        for missing_key in missing_keys:
            if missing_key not in task:
                task[missing_key] = None
    
    return task_list


def get_leaf_intentions(projects):
    for goal in projects:
        tasks = []
        
        item_queue = deque(goal["ch"])
        
        while len(item_queue) > 0:
            task = item_queue.popleft()

            # If the task has no children tasks (i.e. it is a leaf node)
            if "ch" not in task.keys() or len(task["ch"]) == 0:
                tasks.append(task)
            else:
                item_queue.extend(task["ch"])
        
        goal["ch"] = tasks
    
    return projects


def get_tag_regex(tag):
    return fr"#{tag}(?:\b|)"


def misc_tasks_to_goals(real_goals, misc_goals, extra_time=0):
    """
    Converts misc-goal tasks into goals for themselves. That is, each task is a
    goal for itself consisting of only one task (itself).
    
    Args:
        real_goals: [Goal] representing real goals
        misc_goals: [{node}] representing misc goals
        extra_time: Additional (in minutes) time that shifts the deadline

    Returns:
        [Task]
    """
    
    # Initialize latest deadline to 0 (in case no real goals are provided)
    latest_deadline = 0
    
    if len(real_goals) > 0:
        # Sort goals
        real_goals.sort()
        
        # Get latest deadline of real goals
        latest_deadline = real_goals[-1].get_latest_deadline_time()
    
    # Update latest deadline
    total_misc_time_est = 0
    
    # Calculate misc goal time estimation & total misc-goal time estimation
    for misc_goal in misc_goals:
        misc_goal["est"] = 0
        
        for misc_task in misc_goal["ch"]:
            misc_goal["est"] += misc_task["est"]
            
        total_misc_time_est += misc_goal["est"]

    # Add fictive deadline for misc goals, i.e. move latest deadline for
    # <total misc-goal time estimation> + <extra time> minutes in the future.
    latest_deadline += total_misc_time_est + extra_time
    
    # Decompose misc goals into goals for each task of the goals
    misc_tasks = deque()
    for misc_goal in misc_goals:
        
        # Assign deadline and value for misc goal
        if (misc_goal["deadline"]) is None:
            misc_goal['deadline'] = latest_deadline

        # Create a goal for each misc-goal task
        for task in misc_goal['ch']:
            
            # Initialize task goal
            task_goal = dict()
            
            for key in misc_goal.keys():
                
                # Copy everything except children nodes
                if key != 'ch':
                    task_goal[key] = misc_goal[key]

            if task["deadline"]:
                task_goal["deadline"] = task["deadline"]

            if task["deadline_datetime"]:
                task_goal["deadline_datetime"] = task["deadline_datetime"]
                
            task_goal["est"] = task["est"]
            task_goal['id'] = task['id']
            task_goal['nm'] = task['nm']
            task_goal["parentId"] = task["parentId"]

            # Calculate linear/fractional task value w.r.t. misc goal value
            task_goal["value"] *= task["est"] / misc_goal["est"]
            
            task_goal["ch"] = [task]

            misc_tasks.append(task_goal)

    return list(misc_tasks)


def parse_current_intentions_list(current_intentions, default_time_est=None):
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
        split = task_name.split("$wf:")
        
        # If there is a WorkFlowy ID included in the intention name
        if len(split) > 1:
            return split[-1]  # Return the WorkFlowy ID
        else:
            return "__no_wf_id__"  # Return dummy WorkFlowy ID

    # Dictionary of all parsed current intentions
    current_intentions_dict = dict()
    
    for task in current_intentions:
        task_dict = dict()
        
        # Get time estimation
        task_dict["est"] = 0
        
        # Check whether current intention has been "neverminded"
        task_dict["nvm"] = False
        if "nvm" in task.keys():
            task_dict["nvm"] = task["nvm"]
        
        # If current intention is still active
        if not task_dict["nvm"]:
            hours = re.search(HOURS_REGEX, task["t"], re.IGNORECASE)
            if hours is not None:
                hours = hours[0].strip()
                hours = int(hours.split(" ")[0].strip())
                task_dict["est"] += hours * 60
    
            minutes = re.search(MINUTES_REGEX, task["t"], re.IGNORECASE)
            if minutes is not None:
                minutes = minutes[0].strip()
                minutes = int(minutes.split(" ")[0].strip())
                task_dict["est"] += minutes
                
        # Get other necessary information
        task_dict["id"] = get_wf_task_id(task["t"])
        task_dict["d"] = task["d"] if "d" in task.keys() else False
        task_dict["vd"] = task["vd"] if "vd" in task.keys() else None
        
        # Add current task to the dictionary of all parsed current intentions
        current_intentions_dict.setdefault(task_dict["id"], [])
        current_intentions_dict[task_dict["id"]].append(task_dict)
        
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
    return int(re.search(TOTAL_VALUE_REGEX, time_string, re.IGNORECASE)[1])
    

def parse_tree(projects, current_intentions, today_minutes, typical_minutes,
               default_deadline, min_sum_of_goal_values,
               max_sum_of_goal_values, min_goal_value_per_goal_duration,
               max_goal_value_per_goal_duration, time_zone):
    """
    This function reads in a flattened project tree and parses fields like goal
    code, total value, duration and deadline
    """
    def get_wf_task_id(task_name):
        return task_name.split("-")[-1]
    
    # Initialize lists of real and miscellaneous goals
    real_goals = []
    misc_goals = []
    
    # Initialize sum of goal values
    sum_of_goal_values = 0
    
    for goal in projects:
        
        # Extract goal information
        goal["code"] = re.search(INPUT_GOAL_CODE_REGEX, goal["nm"],
                                 re.IGNORECASE)[1]
        goal_deadline = re.search(DEADLINE_REGEX, goal["nm"], re.IGNORECASE)

        # Process goal deadline and check whether the value is valid
        try:
            goal["deadline"], goal["deadline_datetime"] = \
                process_deadline(goal_deadline, today_minutes,
                                 typical_minutes, time_zone, default_deadline)
        except Exception as error:
            raise Exception(f"Goal {goal['nm']}: {str(error)}")

        # Process goal value and check whether the value is valid
        try:
            goal["value"] = process_goal_value(goal)
            sum_of_goal_values += goal["value"]
        except Exception as error:
            raise Exception(f"Goal {goal['nm']}: {str(error)}")
        
        # Initialize dict of all task deadlines
        goal["task_deadlines"] = dict()
        
        # If the goal code is not a digit --> misc goal
        if goal["code"][0] not in digits+"^":
            if "_CSC209" in goal["nm"]:
                goal["code"] = "💻"
            else:
                goal["code"] = "&"
            misc_goals += [goal]
        else:
            real_goals += [goal]
            
        for task in goal["ch"]:
            
            # Get the last part of the HEX ID code for the task in WorkFlowy
            task_id = get_wf_task_id(task["id"])
            
            # Get task deadline (if provided)
            task_deadline = re.search(DEADLINE_REGEX, task["nm"], re.IGNORECASE)
            
            if task_deadline:
                try:
                    task["deadline"], task["deadline_datetime"] = \
                        process_deadline(task_deadline, today_minutes,
                                         typical_minutes, time_zone)
                except Exception as error:
                    raise Exception(f"Task {task['nm']}: {str(error)}")

                # Check whether task deadline is after goal deadline
                if task["deadline"] > goal["deadline"]:
                    raise Exception(f"Task {task['nm']}: Task deadline should "
                                    f"be before goal's deadline.")
                
                # TODO: Add comments...
                str_deadline = str(task["deadline"])  # MongoDB requirement!
                goal["task_deadlines"].setdefault(str_deadline, 0)
                goal["task_deadlines"][str_deadline] += task["est"]
                
            else:
                task["deadline"] = None
                task["deadline_datetime"] = None
                
                # TODO: Add comments...
                str_deadline = str(goal["deadline"])  # MongoDB requirement!
                goal["task_deadlines"].setdefault(str_deadline, 0)
                goal["task_deadlines"][str_deadline] += task["est"]
                
            # Check whether the task has already been scheduled in CompliceX or
            # completed in WorkFlowy
            if task_id in current_intentions.keys() or \
                    ("cp" in task.keys() and task["cp"] >= task["lm"]):
                task["completed"] = True
            else:
                task["completed"] = False
                
            # Check whether a specific date is given
            task['day_datetime'] = process_working_date(task)
    
            # Check whether a task has been marked to be completed in the future
            task["future"] = process_tagged_item("future", task)
            
            # Check whether a task has been marked to be completed today
            task["today"] = process_tagged_item("today", task)
            
            task["parentId"] = goal["id"]
            task["pcp"] = False  # TODO: Not sure what this field is...
            
            # Append goal's name to task's name
            task["nm"] = goal["code"] + ") " + task["nm"]
            
            # Assign points per hour
            task["pph"] = goal["value"] / goal["est"] * 60

        # Set latest start time
        # goal["latest_start_time"] = compute_latest_start_time(goal)

        # Set estimated goal deadline
        # goal["effective_deadline"] = goal["latest_start_time"] + goal["est"]
        
        # Check goal value per duration
        value_per_duration = goal["value"] / goal["est"]
        if min_goal_value_per_goal_duration != float('inf') and \
                max_goal_value_per_goal_duration != float('inf') and not \
                min_goal_value_per_goal_duration <= value_per_duration <= max_goal_value_per_goal_duration:
            raise Exception(f"Goal {goal['nm']} has value per duration of "
                            f"{value_per_duration:.2f} and it should be in the "
                            f"range between {min_goal_value_per_goal_duration:.2f} "
                            f"and {max_goal_value_per_goal_duration:.2f}."
                            f"Please change your goal values.")

    # Check goal value per duration
    if min_sum_of_goal_values != float('inf') and \
            max_sum_of_goal_values != float('inf') and not \
            min_sum_of_goal_values <= sum_of_goal_values <= max_sum_of_goal_values:
        raise Exception(f"Your goals have total values of {sum_of_goal_values} "
                        f"and this value should be in the range between "
                        f"{min_sum_of_goal_values:.2f} and "
                        f"{max_sum_of_goal_values:.2f}. "
                        f"Please change your goal values.")

    return real_goals, misc_goals


def process_deadline(deadline, today_minutes, typical_minutes, time_zone,
                     default_deadline=None):
    def time_delta_to_minutes(time_delta):
        return time_delta.days * 24 * 60 + time_delta.seconds // 60
    
    # Set starting time to the UTC time at the moment
    current_time = datetime.utcnow()
    
    # Shift the starting time according to the time zone
    current_time += timedelta(minutes=time_zone)
    
    # If no deadline provided, set the default deadline
    if deadline is None:
        if default_deadline is not None:
            default_deadline_datetime = \
                timedelta(days=int(default_deadline))
            deadline = \
                re.search(DEADLINE_REGEX, "DUE:" +
                          (current_time + default_deadline_datetime).strftime("%Y-%m-%d"),
                          re.IGNORECASE)
        else:
            raise Exception("Invalid or no deadline provided!")
    
    deadline_datetime = date_str_to_datetime(deadline)
    time_delta = deadline_datetime - current_time
    regular_deadline_minutes = time_delta_to_minutes(time_delta)

    # Calculate the number of days until the deadline
    days_after_today = (deadline_datetime.date() - current_time.date()).days
    
    # Calculate the number of weeks until the deadline
    weeks_after_today = days_after_today // 7
    
    # Get information on the weekdays of the current day and the deadline day
    current_weekday = current_time.weekday()
    
    # Initialize number of minutes after today's day
    minutes_after_today = 0
    
    # Add available time w.r.t. number of weeks until deadline
    for day_idx in range(7):
        minutes_after_today += typical_minutes[day_idx] * weeks_after_today
        
    # Add available time w.r.t. remainder days
    for day in range(days_after_today % 7):
        weekday = (current_weekday + day + 1) % 7
        minutes_after_today += typical_minutes[weekday]

    # Calculate how much time is left today & update today's minutes
    end_of_the_day = datetime(current_time.year, current_time.month,
                              current_time.day, 23, 59, 59)
    minutes_left_today = (end_of_the_day - current_time).seconds // 60
    today_minutes = min(today_minutes, minutes_left_today,
                        regular_deadline_minutes)
    
    # Calculate deadline value
    deadline_value = today_minutes + minutes_after_today

    # Check whether it is in the future
    if deadline_datetime < current_time:
        raise Exception(f"Deadline not in the future! Please check your "
                        f"deadline and your time zone.")

    return deadline_value, deadline_datetime


def process_goal_value(goal):
    goal_value = re.search(TOTAL_VALUE_REGEX, goal["nm"], re.IGNORECASE)
    
    if goal_value is None:
        raise Exception("No value or invalid value provided!")
    
    # Parse value
    goal_value = int(goal_value[1])

    # Check whether it is a positive number
    if goal_value <= 0:
        raise Exception("Value not a positive number!")

    return goal_value


def process_tagged_item(tag, task):
    tag_regex = get_tag_regex(tag)
    tag_present = re.search(tag_regex, task["nm"].lower(), re.IGNORECASE)
    
    if tag_present:  # ... is not None
        return True
    else:
        return False


def process_time_est(task_name, allowed_task_time=float('inf'),
                     default_time_est=None):
    try:
        time_est = re.search(TIME_EST_REGEX, task_name, re.IGNORECASE)[0]
    except:
        if default_time_est is not None:
            time_est = "~~" + str(default_time_est) + "min"
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


def process_task_days(task):
    weekdays = [False for _ in range(7)]  # Monday (0) to Sunday (6)
    repetitive_weekdays = [False for _ in range(7)]  # Monday (0) to Sunday (6)
    
    # Check individual weekdays
    for day_idx, day in enumerate(WEEKDAYS.values()):
        if process_tagged_item(day.lower(), task):
            weekdays[day_idx] = True
        if process_tagged_item(day.lower() + 's', task):
            weekdays[day_idx] = True
            repetitive_weekdays[day_idx] = True
            
    # Check #weekdays
    if process_tagged_item('weekdays', task):
        for day_idx in [0, 1, 2, 3, 4]:  # Monday to Friday
            weekdays[day_idx] = True
            repetitive_weekdays[day_idx] = True

    # Check #weekends
    if process_tagged_item('weekends', task):
        for day_idx in [5, 6]:  # Saturday and Sunday
            weekdays[day_idx] = True
            repetitive_weekdays[day_idx] = True

    return weekdays, repetitive_weekdays


def process_working_date(task):
    date_datetime = None
    
    # Standardize input
    task_name = task['nm'].lower()
    
    # Search for #<date>
    date = re.search(fr"#\s*{DATE_REGEX}", task_name, re.IGNORECASE)
    
    # If #<date> is found
    if date:
        date_datetime = date_str_to_datetime(date)
    
    return date_datetime


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
                 # effective_deadline=goal["effective_deadline"],
                 # latest_start_time=goal["latest_start_time"],
                 rewards={goal["deadline"]: goal["value"]},
                 penalty=0))  # TODO: Penalty for missing a deadline
        
    return goals
