from collections import deque

from pprint import pprint
from src.schedulers.helpers import *
from src.utils import task_dict_from_projects


def basic_scheduler(task_list, current_day, duration_remaining=8 * 60,
                    with_today=True):
    """
    Takes in flattened project tree with "reward" from some API
    Outputs list of tasks for today
    """

    # Initialize queue of tasks for today
    today_tasks = deque()

    # Initialize queue of other tasks eligible to be scheduled today
    remaining_tasks = deque()

    # Get information on current weekday
    current_weekday = current_day.weekday()

    if with_today:
        for task in task_list:
    
            # If the task is not completed and not indefinitely postponed (future)
            if not task["completed"] and not task["future"]:
        
                # If task is marked to be scheduled today by the user
                if task["scheduled_today"]:
                    today_tasks.append(task)
                    duration_remaining -= task["est"]
        
                # If task is should be repetitively scheduled on today's day
                elif is_repetitive_task(task, weekday=current_weekday):
                    today_tasks.append(task)
                    duration_remaining -= task["est"]
        
                # If the task is eligible to be scheduled today
                elif check_additional_scheduling(
                        task, current_day, current_weekday):
                    remaining_tasks.append(task)
        
    # From: https://stackoverflow.com/a/73050
    sorted_by_deadline = sorted(list(remaining_tasks),
                                key=lambda k: k['deadline'])
    for task in sorted_by_deadline:
        
        # If not time left, don't add additional tasks (without #today)
        if duration_remaining == 0:
            break

        # If there is enough time to schedule task
        if task["est"] <= duration_remaining:
            today_tasks.append(task)
            duration_remaining -= task["est"]

    return list(today_tasks)


def deadline_scheduler(task_list, current_day, deadline_window=1,
                       today_duration=8 * 60, with_today=True):
    # Tasks within deadline window are tagged with today
    for task in task_list:
        if task["deadline"] <= deadline_window:
            task["today"] = True
    
    final_tasks = basic_scheduler(task_list, current_day=current_day,
                                  duration_remaining=today_duration,
                                  with_today=with_today)
    return final_tasks


def schedule_tasks_for_today(projects, ordered_tasks, duration_remaining,
                             current_day):
    
    # Get task dictionary from JSON tree
    task_dict = task_dict_from_projects(projects)
    
    # Initialize queue of tasks for today
    today_tasks = deque()
    
    # Initialize queue of other tasks eligible to be scheduled today
    remaining_tasks = deque()
    
    # Get information on current weekday
    current_weekday = current_day.weekday()

    for task in ordered_tasks:
        
        task_id = task.get_id()
        task_item = task_dict[task_id]
        
        task_item["est"] = task.get_time_est()
        task_item["val"] = task.get_optimal_reward()

        # If the task is not completed and not indefinitely postponed (future)
        if not task_item["completed"] and not task_item["future"]:
            
            # If task is marked to be scheduled today by the user
            if task_item["scheduled_today"]:
                today_tasks.append(task_item)
                duration_remaining -= task_item["est"]

            # If task is should be repetitively scheduled on today's day
            elif is_repetitive_task(task_item, weekday=current_weekday):
                today_tasks.append(task_item)
                duration_remaining -= task_item["est"]

            # If the task is eligible to be scheduled today
            elif check_additional_scheduling(task_item,
                                             current_day, current_weekday):
                remaining_tasks.append(task_item)
                
    # Schedule other tasks if time left
    while len(remaining_tasks) > 0 and duration_remaining > 0:

        # Get next task in the list
        task_item = remaining_tasks.popleft()
        
        # If there is enough time to schedule task
        if task_item["est"] <= duration_remaining:
            today_tasks.append(task_item)
            duration_remaining -= task_item["est"]
        
    today_tasks = list(today_tasks)
    today_tasks.sort(key=lambda task: -task["val"])
    
    return today_tasks
