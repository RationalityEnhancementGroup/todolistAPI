from datetime import datetime, timedelta

from todolistAPI.src.utils import task_dict_from_projects


def basic_scheduler(task_list, time_zone=0, today_duration=8 * 60,
                    with_today=True):
    """
    Takes in flattened project tree with "reward" from some API
    Outputs list of tasks for today
    """
    duration_remaining = today_duration
    
    final_tasks = []
    # overwork_minutes = 0
    
    if with_today:
        for task in task_list:
            
            # If the task has not been completed and it marked for today
            if check_priority_scheduling(task, time_zone=time_zone):
                
                # Schedule task
                final_tasks.append(task)
                duration_remaining -= task["est"]
                
                # If there is enough time to schedule the task
                # if task["est"] <= duration_remaining:
                #     final_tasks.append(task)
                #     duration_remaining -= task["est"]
                # else:
                #     overwork_minutes += task["est"]
    
    # overwork_minutes -= duration_remaining
    # if overwork_minutes > 0:
    #     raise Exception(generate_overwork_error_message(overwork_minutes))
    
    # From: https://stackoverflow.com/a/73050
    sorted_by_deadline = sorted(task_list, key=lambda k: k['deadline'])
    for task in sorted_by_deadline:
        
        # If not time left, don't add additional tasks (without #today)
        if duration_remaining == 0:
            break
        
        if check_additional_scheduling(task, duration_remaining,
                                       time_zone=time_zone):
            final_tasks.append(task)
            duration_remaining -= task["est"]
    
    return final_tasks


def check_additional_scheduling(task, duration_remaining, time_zone=0):
    return (task["est"] <= duration_remaining) and not \
        (task["completed"] or task["future"]) and \
        check_date_assignment(task, time_zone=time_zone) is None and \
        check_today_assignment(task, time_zone=time_zone) is None


def check_date_assignment(task, time_zone=0):
    current_day = datetime.utcnow() + timedelta(minutes=time_zone)

    if task['day_datetime'] is None:
        return None
    
    return task['day_datetime'].date() == current_day.date()


def check_priority_scheduling(task, time_zone=0):
    return not (task["completed"] or task["future"]) and \
           (check_date_assignment(task, time_zone=time_zone) or
            check_today_assignment(task, time_zone=time_zone))


def check_today_assignment(task, time_zone=0):
    # Check whether the task has to be scheduled today
    if task["daily"] or task["today"]:
        return True
    
    current_day = datetime.utcnow() + timedelta(minutes=time_zone)
    current_weekday = current_day.weekday()
    
    # Check whether there is a weekday preference
    if task["task_days"][current_weekday]:
        return True

    # Check whether there are weekday preferences
    # - None: There are no weekday preferences
    # - False: There are weekday preferences, but not for today
    weekday_assignment = None
    for day_idx in range(len(task["task_days"])):
        if task["task_days"][day_idx]:
            weekday_assignment = False
            
    # Return whether there are weekday preferences
    return weekday_assignment


def deadline_scheduler(task_list, deadline_window=1, time_zone=0,
                       today_duration=8 * 60, with_today=True):
    # Tasks within deadline window are tagged with today
    for task in task_list:
        if task["deadline"] <= deadline_window:
            task["today"] = True
    
    final_tasks = basic_scheduler(task_list, time_zone=time_zone,
                                  today_duration=today_duration,
                                  with_today=with_today)
    return final_tasks


def generate_overwork_error_message(overwork_minutes):
    return f"You have scheduled {overwork_minutes} additional minutes of " \
           f"work for today. Please change your HOURS_TODAY value in the " \
           f"WorkFlowy tree or reduce the amount of work by removing #daily " \
           f"or #today for some of the tasks."


def schedule_tasks_for_today(projects, ordered_tasks, duration_remaining,
                             time_zone=0):
    task_dict = task_dict_from_projects(projects)
    
    today_tasks = []
    # overwork_minutes = 0
    
    for task in ordered_tasks:
        task_id = task.get_task_id()
        task_item = task_dict[task_id]
        task_item["val"] = task.get_reward()
        
        # If the task has not been completed and it is marked for today
        if check_priority_scheduling(task_item, time_zone=time_zone):
    
            # Schedule task
            today_tasks += [task_item]
            duration_remaining -= task_item["est"]
            
            # If there is enough time to schedule the task
            # if duration_remaining >= task_item["est"]:
            #     today_tasks += [task_item]
            #     duration_remaining -= task_item["est"]
            # else:
            #     overwork_minutes += task_item["est"]

    # overwork_minutes -= duration_remaining
    # if overwork_minutes > 0:
    #     raise Exception(generate_overwork_error_message(overwork_minutes))
    
    # Schedule other tasks
    for task in ordered_tasks:
        
        # If not time left, don't add additional tasks (without #today)
        if duration_remaining == 0:
            break
        
        task_id = task.get_task_id()
        task_item = task_dict[task_id]
        
        # If the task has not been completed and it is not for today and
        # there is enough time to complete the task today
        if check_additional_scheduling(task_item, duration_remaining,
                                       time_zone=time_zone):
            today_tasks += [task_item]
            duration_remaining -= task_item["est"]
    
    return today_tasks
