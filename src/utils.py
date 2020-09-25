import cherrypy
import numpy as np
import re

from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta
from math import ceil
from string import digits

from todolistMDP.to_do_list import Item

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


def compute_latest_start_time(goal):
    # Initialize current time
    current_time = goal["deadline"]
    
    # Iterate tasks in reverse order w.r.t. deadline
    for task in reversed(goal["ch"]):
        current_time = min(current_time, task["deadline"])
        current_time -= int(task["est"])
    
        if current_time < 0:
            raise Exception(f'Task {task["nm"]} is unattainable! '
                            f'Please reschedule its deadline.')
    
    return current_time


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


def delete_sensitive_data(projects):
    
    def recursive_deletion(super_item):
        
        # Delete item name
        del super_item["nm"]
        
        # TODO: What is "no"
        try:
            del super_item["no"]
        except:
            pass

        # Delete sensitive data recursively (if item has children nodes)
        if "ch" in super_item.keys():
            
            for item in super_item["ch"]:
                
                recursive_deletion(item)

    # Make deep copy of the complete data dictionary
    items = deepcopy(projects)

    # Delete sensitive data recursively
    for item in items:
        
        recursive_deletion(item)
        
    return items


def flatten_intentions(projects):
    for goal in projects:
        for task in goal["ch"]:
            if "ch" in task:
                goal["ch"].extend(task["ch"])
                del task["ch"]
    return projects


def get_final_output(item_list, round_param, points_per_hour, user_datetime):
    """
    Input is list of items
    Outputs list of items for today with fields:
        id, nm, lm, parentId, pcp, est, val (=reward)
    """
    
    def get_human_readable_name(item):
        item_name = item["nm"]
        
        # Remove #date regex
        item_name = re.sub(fr"#\s*{DATE_REGEX}", "", item_name, re.IGNORECASE)

        # Remove deadline
        item_name = re.sub(DEADLINE_REGEX, "", item_name, re.IGNORECASE)
        
        # Remove time estimation
        item_name = re.sub(TIME_EST_REGEX, "", item_name, re.IGNORECASE)
        
        # Remove tags
        for tag in TAGS:
            tag_regex = get_tag_regex(tag)
            item_name = re.sub(tag_regex, "", item_name, re.IGNORECASE)
        
        item_name = item_name.strip()
        
        if len(re.sub(OUTPUT_GOAL_CODE_REGEX, "", item_name).strip()) == 0:
            raise NameError(f"Item {item['nm']} has no name!")
        
        # Append time information
        hours, minutes = item["est"] // 60, item["est"] % 60
        
        item_name += " (takes about "
        if hours > 0:
            if hours == 1:
                item_name += f"1 hour"
            else:
                item_name += f"{hours} hours"
        if minutes > 0:
            if hours > 0:
                item_name += " and "
            if minutes == 1:
                item_name += f"1 minute"
            else:
                item_name += f"{minutes} minutes"

        if hasattr(item, "deadline_datetime") and \
                item["deadline_datetime"] is not None:
            item_name += ", due on "

            td = item["deadline_datetime"] - user_datetime
            if td.days < 7:
                weekday = item["deadline_datetime"].weekday()
                item_name += WEEKDAYS[weekday]
            else:
                item_name += str(item["deadline_datetime"])[:-3]
        
        item_name += ")"
        
        return item_name
    
    keys_needed = ["id", "nm", "lm", "parentId", "pcp", "est", "val"]
    
    # for now only look at first dictionary
    current_keys = set(item_list[0].keys())
    extra_keys = list(current_keys - set(keys_needed))
    missing_keys = list(set(keys_needed) - current_keys)
    
    for item in item_list:
        
        item["nm"] = get_human_readable_name(item)
        
        if points_per_hour:
            item["val"] = str(round(item["pph"], round_param)) + '/h'
        else:
            item["val"] = round(item["val"], round_param)

        for extra_key in extra_keys:
            if extra_key in item:
                del item[extra_key]
        
        for missing_key in missing_keys:
            if missing_key not in item:
                item[missing_key] = None
    
    return item_list


def get_leaf_intentions(projects):
    for goal in projects:
        
        tasks = deque()
        
        item_queue = deque(goal["ch"])
        
        while len(item_queue) > 0:
            item = item_queue.popleft()

            # If item has no children items (i.e. it is a leaf node)
            if "ch" not in item.keys() or len(item["ch"]) == 0:
                tasks.append(item)
            
            # If item is an intermediate node
            else:
                item_queue.extend(item["ch"])
        
        goal["ch"] = list(tasks)
    
    return projects


def get_tag_regex(tag):
    return fr"#{tag}(?:\b|)"


def get_wf_item_id(item_name):
    """
    Extracts the WorkFlowy ID from the name of the item.
    Args:
        item_name: Item name

    Returns:
        Item ID
    """
    split = item_name.split("$wf:")
    
    # If there is a WorkFlowy ID included in the intention name
    if len(split) > 1:
        return split[-1]  # Return the WorkFlowy ID
    else:
        return "__no_wf_id__"  # Return dummy WorkFlowy ID


def generate_to_do_list(projects, allowed_task_time, available_time,
                        current_intentions, default_deadline, default_time_est,
                        planning_fallacy_const, user_datetime,
                        min_sum_of_goal_values=np.NINF,
                        max_sum_of_goal_values=np.PINF,
                        min_goal_value_per_duration=np.NINF,
                        max_goal_value_per_duration=np.PINF):
    # TODO: Change function input to be parameters (dict)
    
    # Initialize sum of goal values
    sum_of_goal_values = 0
    
    for goal in projects:
        
        """ ===== Initialization =====
            - Goal code
            - Goal value
        """
        
        # Remove formatting / HTML formatting
        goal["nm"] = re.sub(HTML_REGEX, "", goal["nm"],
                            count=LARGE_NUMBER, flags=re.IGNORECASE)
        
        # Extract goal code
        goal["code"] = re.search(INPUT_GOAL_CODE_REGEX, goal["nm"],
                                 re.IGNORECASE)[1]
        
        if goal["code"][0] not in digits + "^":
            if "_CS" in goal["nm"]:
                goal["code"] = "💻"
            else:
                goal["code"] = "&"
        
        # Initialize completed
        goal["completed"] = None
        
        # Process goal value and check whether the value is valid
        try:
            goal["value"] = process_goal_value(goal)
        except Exception as error:
            raise Exception(f"Goal {goal['nm']}: {str(error)}")
        
        # Update sum of goal values
        sum_of_goal_values += goal["value"]
        
        # # Initialize dict of all task deadlines
        # goal["task_deadlines"] = dict()
        
        # # Set latest start time
        # goal["latest_start_time"] = compute_latest_start_time(goal)
        
        # # Set estimated goal deadline
        # goal["effective_deadline"] = goal["latest_start_time"] + goal["est"]
        
        """ ===== 1st traversal =====
            - Passing down
                - Goal code
                - Parent ID
                - Scheduling tags (?)
                - Value of sub-goals (?)

            - Initialize
                - Item completed
                - Item name with goal code up front
                - Parent completed
                - Scheduling tags
                - Time estimate
                - Time allocated for today

            - Passing up
                - Total time estimate for children nodes
                - Time allocated for today (inplace update)
                - Time allocated for recurring tasks (inplace update)
        """
        
        # Traverse sub-tree
        first_traversal(super_item=goal,
                        allowed_task_time=allowed_task_time,
                        available_time=available_time,
                        current_intentions=current_intentions,
                        default_time_est=default_time_est,
                        planning_fallacy_const=planning_fallacy_const,
                        user_datetime=user_datetime)
        
        # Check goal value per duration
        value_per_duration = goal["value"] / goal["est"]
        if not (
                min_goal_value_per_duration <= value_per_duration <= max_goal_value_per_duration):
            raise Exception(f"Goal {goal['nm']} has value per duration of "
                            f"{value_per_duration:.2f} and it should be in the "
                            f"range between {min_goal_value_per_duration:.2f} "
                            f"and {max_goal_value_per_duration:.2f}."
                            f"Please change your goal values.")
        
        # Subtract time estimate of current intentions from available time
        # TODO: Check whether it works properly (!)
        for tasks in current_intentions.values():
            for task in tasks:
                
                # If the task is not marked as completed or "nevermind"
                if not task["d"] and not task["nvm"]:
                    available_time[0] -= task["est"]
        
        # Make 0 if the number of minutes is negative
        available_time[0] = max(available_time[0], 0)
        
        """ ===== 2nd traversal =====
            Pass down:
                - Deadline (if not assigned)
                - Deadline datetime (if not assigned)
                - Value proportion
                
            Initialize:
                - Deadline
                - Deadline datetime
                - Point per hour
        """
        # Extract goal deadline
        goal_deadline = re.search(DEADLINE_REGEX, goal["nm"], re.IGNORECASE)
        
        # Process goal deadline and check whether the value is valid
        try:
            goal["deadline"], goal["deadline_datetime"] = \
                process_deadline(goal_deadline,
                                 current_datetime=user_datetime,
                                 default_deadline=default_deadline,
                                 today_minutes=available_time[0],
                                 typical_minutes=available_time[1:])
        except Exception as error:
            raise Exception(f"Goal {goal['nm']}: {str(error)}")

        second_traversal(goal,
                         today_minutes=available_time[0],
                         typical_minutes=available_time[1:],
                         user_datetime=user_datetime)
        
    # Check goal value per duration
    if not (min_sum_of_goal_values <= sum_of_goal_values <= max_sum_of_goal_values):
        raise Exception(f"Your goals have total values of {sum_of_goal_values} "
                        f"and this value should be in the range between "
                        f"{min_sum_of_goal_values:.2f} and "
                        f"{max_sum_of_goal_values:.2f}. "
                        f"Please change your goal values.")
    
    return projects


def first_traversal(super_item, allowed_task_time, available_time,
                    current_intentions, default_time_est,
                    planning_fallacy_const, user_datetime):
    
    # Initialize goal time estimate
    super_item["est"] = 0
    
    for item in super_item["ch"]:
        
        # Remove formatting / HTML tags
        item["nm"] = re.sub(HTML_REGEX, "", item["nm"],
                            count=LARGE_NUMBER, flags=re.IGNORECASE)
        
        # Default initialization
        item["code"] = super_item["code"]
        item["parentId"] = super_item["id"]
        item["pcp"] = False  # pcp: Parent completed
        
        item["completed"] = None
        item["days"] = [None] * 7
        item["day_datetime"] = None
        item["daily"] = None
        item["future"] = None
        item["repetitive_days"] = [None] * 7
        item["scheduled_today"] = None
        
        # If the sub-item is not a leaf node (i.e. sub-goal)
        if "ch" in item.keys() and len(item["ch"]) > 0:
            
            # Traverse sub-trees
            first_traversal(super_item=item,
                            current_intentions=current_intentions,
                            allowed_task_time=allowed_task_time,
                            available_time=available_time,
                            default_time_est=default_time_est,
                            planning_fallacy_const=planning_fallacy_const,
                            user_datetime=user_datetime)
        
        # If the sub-item is a leaf node (i.e. task)
        else:
            
            """ Process time estimate """
            try:
                # Parse task time estimate
                item["est"] = \
                    process_time_est(item["nm"],
                                     allowed_task_time, default_time_est)
                
                # Apply planning fallacy
                item["est"] = int(ceil(item["est"] * planning_fallacy_const))
            
            except Exception as error:
                raise Exception(f"Task {item['nm']}: {str(error)}")
            
            """ Process scheduling tags """
            # TODO: Not sure whether this should apply to intermediate nodes
            
            # Check whether weekday preferences are given
            item["days"], item["repetitive_days"] = process_task_days(item)
            
            # Check whether it is a daily task
            item["daily"] = process_tagged_item("daily", item)
            
            if item["daily"]:
                for weekday in range(len(item["repetitive_days"])):
                    item["repetitive_days"][weekday] = True
                    
                    # Subtract time estimate from typical working time
                    available_time[weekday + 1] -= item["est"]
            
            # Subtract busy time from each scheduled weekday
            for weekday, repetitive in enumerate(item["repetitive_days"]):
                if repetitive:
                    available_time[weekday + 1] -= item["est"]
            
            # Check whether a specific date is given
            item['day_datetime'] = process_working_date(item)
            
            # Check whether a task has been marked to be completed in the future
            item["future"] = process_tagged_item("future", item)
            
            # Check whether a task has been marked to be completed today
            item["today"] = process_tagged_item("today", item)
            
            # Check whether the task should be scheduled today (w/o repetition!)
            weekday = user_datetime.date().weekday()
            
            item["scheduled_today"] = (
                    item["today"] or item["days"][weekday] or
                    (item["day_datetime"] is not None and
                     item["day_datetime"].date() == user_datetime.date())
            )
            
            # Subtract the time estimate needed to be scheduled today
            if item["scheduled_today"]:
                available_time[0] -= item["est"]
            
            """ Check whether task has already been completed """
            # Get the last part of the HEX ID code for the task in WorkFlowy
            item_id = item["id"].split("-")[-1]
            
            # Check whether the task has already been scheduled or completed
            if item_id in current_intentions.keys() or \
                    ("cp" in item.keys() and item["cp"] >= item["lm"]):
                item["completed"] = True
            else:
                item["completed"] = False
                
        # Append goal's name to task's name
        item["nm"] = super_item["code"] + ") " + item["nm"]

        # Update time estimate
        if not item["completed"]:
            super_item["est"] += item["est"]

    return


def second_traversal(super_item, today_minutes, typical_minutes, user_datetime):
    
    # Initialize sanity check
    total_value = 0
    
    for item in super_item["ch"]:
    
        # Assign points per hour
        item["pph"] = super_item["value"] / super_item["est"] * 60
        
        """ Process deadline """
        # Get item deadline (if provided)
        deadline = re.search(DEADLINE_REGEX, item["nm"], re.IGNORECASE)
        
        if deadline:
            try:
                item["deadline"], item["deadline_datetime"] = \
                    process_deadline(deadline,
                                     current_datetime=user_datetime,
                                     # default_deadline=default_deadline,
                                     today_minutes=today_minutes,
                                     typical_minutes=typical_minutes)
            
            except Exception as error:
                raise Exception(f"Item {item['nm']}: {str(error)}")
            
            # Check whether item deadline is after goal deadline
            if item["deadline"] > super_item["deadline"]:
                raise Exception(f"Item {item['nm']}: "
                                f"Item deadline "
                                f"{item['deadline_datetime']} "
                                f"should be before (sub-)goal's deadline "
                                f"{super_item['deadline_datetime']}.")
        
        # If item has no deadline, set its deadline to be (sub-)goal deadline
        else:
            item["deadline"] = super_item["deadline"]
            item["deadline_datetime"] = super_item["deadline_datetime"]
        
        # Compute sub-tree value
        item["value"] = item["est"] / super_item["est"] * super_item["value"]
        
        # Update total value
        if not item["completed"]:
            total_value += item["value"]
        
        # Traverse sub-tree recursively
        if "ch" in item.keys() and len(item["ch"]) > 0:
            second_traversal(item,
                             today_minutes=today_minutes,
                             typical_minutes=typical_minutes,
                             user_datetime=user_datetime)
    
    # Sanity check
    assert np.isclose(super_item["value"], total_value, atol=1e-6)
    
    return


def parse_current_intentions_list(current_intentions):
    """
    Extracts necessary information from CompliceX's current intentions list.
    
    Args:
        current_intentions: List of current intentions on CompliceX.

    Returns:
        Dictionary of all parsed current intentions.
    """
    # Dictionary of all parsed current intentions
    current_intentions_dict = dict()
    
    for task in current_intentions:
        task_dict = dict()
        
        # Get time estimation
        task_dict["est"] = 0

        # Check whether current intention has been completed
        task_dict["d"] = False
        if "d" in task.keys():
            task_dict["d"] = task["d"]

        # Check whether current intention has been "neverminded"
        task_dict["nvm"] = False
        if "nvm" in task.keys():
            task_dict["nvm"] = task["nvm"]
            
        # Get time estimation
        hours = re.search(HOURS_REGEX, task["t"], re.IGNORECASE)
        if hours is not None:
            hours = hours[0].strip()
            hours = float(hours.split(" ")[0].strip())
            task_dict["est"] += hours * 60

        minutes = re.search(MINUTES_REGEX, task["t"], re.IGNORECASE)
        if minutes is not None:
            minutes = minutes[0].strip()
            minutes = int(minutes.split(" ")[0].strip())
            task_dict["est"] += minutes
                
        # Parse WF ID from the task name
        task_dict["id"] = get_wf_item_id(task["t"])
        
        # Get task value in the current intentions list
        task_dict["vd"] = None
        if "vd" in task.keys():
            task_dict["vd"] = task["vd"]
        
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


def process_deadline(deadline, today_minutes, typical_minutes, current_datetime,
                     default_deadline=None):
    
    def time_delta_to_minutes(time_delta):
        return time_delta.days * 24 * 60 + time_delta.seconds // 60
    
    # If no deadline provided, set the default deadline
    if deadline is None:
        if default_deadline is not None:
            default_deadline_datetime = \
                timedelta(days=int(default_deadline))
            deadline = \
                re.search(DEADLINE_REGEX, "DUE:" +
                          (current_datetime + default_deadline_datetime).strftime("%Y-%m-%d"),
                          re.IGNORECASE)
        else:
            raise Exception("Invalid or no deadline provided!")

    # Parse deadline datetime
    deadline_datetime = date_str_to_datetime(deadline)
    
    # Weekday
    weekday = deadline_datetime.weekday()
    
    # Compute time on last day
    minutes_last_day = deadline_datetime.hour * 60 + deadline_datetime.minute
    minutes_last_day = min(minutes_last_day, typical_minutes[weekday])
    
    # Check whether it is in the future
    if deadline_datetime < current_datetime:
        raise Exception(f"Deadline not in the future! Please check your "
                        f"deadline and your time zone.")

    # Compute time difference
    time_delta = deadline_datetime - current_datetime
    
    # Convert time difference to minutes
    regular_deadline_minutes = time_delta_to_minutes(time_delta)

    # Calculate the number of days until the deadline
    days_after_today = (deadline_datetime.date() - current_datetime.date()).days
    
    # Calculate the number of weeks until the deadline
    weeks_after_today = days_after_today // 7
    
    # Get information on the weekdays of the current day and the deadline day
    current_weekday = current_datetime.weekday()
    
    # Initialize number of minutes after today's day
    minutes_after_today = 0
    
    # Add available time w.r.t. number of weeks until deadline
    for day_idx in range(7):
        minutes_after_today += typical_minutes[day_idx] * weeks_after_today
        
    # Add available time w.r.t. remainder days
    for day in range((days_after_today % 7) - 1):
        weekday = (current_weekday + day + 1) % 7
        minutes_after_today += typical_minutes[weekday]

    # Calculate how much time is left today
    end_of_the_day = datetime(current_datetime.year, current_datetime.month,
                              current_datetime.day, 23, 59, 59)
    minutes_left_today = (end_of_the_day - current_datetime).seconds / 60
    minutes_left_today = int(ceil(minutes_left_today))
    
    # Update today minutes
    today_minutes = min(minutes_left_today,
                        min(regular_deadline_minutes, today_minutes))

    # Calculate deadline value
    deadline_value = today_minutes + minutes_after_today
    
    if deadline_datetime.date() != current_datetime.date():
        deadline_value += minutes_last_day

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


def process_tagged_item(tag, item):
    tag_regex = get_tag_regex(tag)
    tag_present = re.search(tag_regex, item["nm"].lower(), re.IGNORECASE)
    
    if tag_present:  # ... is not None
        return True
    else:
        return False


def process_time_est(item_name, allowed_task_time=float('inf'),
                     default_time_est=None):
    try:
        time_est = re.search(TIME_EST_REGEX, item_name, re.IGNORECASE)[0]
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
        raise Exception(f"{item_name}: Time estimation is not a "
                        f"positive number!")
    if duration > allowed_task_time:
        raise Exception(f"{item_name}: Time duration not allowed!")
    
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
            # weekdays[day_idx] = True
            repetitive_weekdays[day_idx] = True
            
    # Check #weekdays
    if process_tagged_item('weekdays', task):
        for day_idx in [0, 1, 2, 3, 4]:  # Monday to Friday
            # weekdays[day_idx] = True
            repetitive_weekdays[day_idx] = True

    # Check #weekends
    if process_tagged_item('weekends', task):
        for day_idx in [5, 6]:  # Saturday and Sunday
            # weekdays[day_idx] = True
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


def item_dict_from_projects(projects):
    return {
        item["id"]: item
        for goal in projects
        for item in goal["ch"]
    }


def task_list_from_projects(projects):
    task_list = []
    for goal in projects:
        for task in goal["ch"]:
            if task["deadline"] is None:
                task["deadline"] = goal["deadline"]
            task_list.append(task)
    return task_list


def tree_to_old_structure(projects, params):
    """
    input: parsed tree
    output: structure that can be inputted to old project code
    """
    def generate_sub_tree(items, goal_item: Item, parent_item=None):

        # Initialize list of items
        item_list = deque()
    
        for item_dict in items:
            
            # Initialize item
            item = Item(
                description=item_dict["nm"],
                
                completed=item_dict["completed"],
                deadline=item_dict["deadline"],
                deadline_datetime=item_dict["deadline_datetime"],
                item_id=item_dict["id"],
                parent_item=parent_item,
                time_est=item_dict["est"],
                today=item_dict["scheduled_today"],
                value=item_dict["value"]
            )

            # If it is an intermediate node
            if "ch" in item_dict.keys() and len(item_dict["ch"]) > 0:
                
                # Generate list of sub-items
                sub_items = generate_sub_tree(
                    items=item_dict["ch"], goal_item=goal_item, parent_item=item
                )
                
                # Add list of sub-items to the parent item
                item.add_items(sub_items)
                
            # If it is a leaf node (i.e. task)
            else:
                
                # Add reference from the goal node to the leaf/task node
                goal_item.append_task(item)
                
                # If task has to be executed today
                if item.is_today() and not item.is_completed():
                    goal_item.today_items.add(item)

            # Compute potential time estimates
            item.compute_binning(num_bins=params["num_bins"])
            
            # Add reference to goal
            item.add_goal(goal_item)

            # Create new item and add it to the item list
            item_list.append(item)

        return item_list
    
    # Initialize list of goals
    goals = deque()
    
    for goal in projects:
    
        # Initialize goal
        goal_item = Item(
            description=goal["nm"],
            
            completed=goal["completed"],
            deadline=goal["deadline"],
            deadline_datetime=goal["deadline_datetime"],
            item_id=goal["id"],
            time_est=goal["est"],
            value=goal["value"]
        )
        
        # Initialize queue of tasks
        goal_item.init_task_list()
        
        # Add list of sub-items
        if "ch" in goal.keys() and len(goal["ch"]) > 0:
            # Generate list of sub-items
            sub_items = generate_sub_tree(items=goal["ch"], goal_item=goal_item,
                                          parent_item=goal_item)
            
            # Add list of sub-items to the parent item
            goal_item.add_items(sub_items)

        # Convert queue of tasks to list
        goal_item.convert_task_list()
        
        # Sort task list
        goal_item.sort_task_list()
            
        # Create new goal and add it to the goal list
        goals.append(goal_item)
        
    return list(goals)
