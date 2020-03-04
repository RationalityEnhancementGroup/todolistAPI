from src.utils import task_dict_from_projects


def basic_scheduler(task_list, today_duration=8 * 60, with_today=True):
    """
    Takes in flattened project tree with "reward" from some API
    Outputs list of tasks for today
    """
    duration_remaining = today_duration
    
    final_tasks = []
    overwork_minutes = 0
    
    if with_today:
        for task in task_list:
            
            # If the task has not been completed and it marked for today
            if not task["completed"] and task["today"]:
    
                # If there is enough time to schedule the task
                if task["est"] <= duration_remaining:
                    final_tasks.append(task)
                    duration_remaining -= task["est"]
                else:
                    overwork_minutes += task["est"]
    
    overwork_minutes -= duration_remaining
    if overwork_minutes > 0:
        raise Exception(generate_overwork_error_message(overwork_minutes))

    # From: https://stackoverflow.com/a/73050
    sorted_by_deadline = sorted(task_list, key=lambda k: k['deadline'])
    for task in sorted_by_deadline:

        # If not time left, don't add additional tasks (without #today)
        if duration_remaining == 0:
            break
            
        if (task["est"] <= duration_remaining) \
                and not task["completed"] \
                and not task["future"] \
                and not task["today"]:
            final_tasks.append(task)
            duration_remaining -= task["est"]
            
    return final_tasks


def deadline_scheduler(task_list, deadline_window=1, today_duration=8 * 60,
                       with_today=True):
    # Tasks within deadline window are tagged with today
    for task in task_list:
        if task["deadline"] <= deadline_window:
            task["today"] = True
    
    final_tasks = basic_scheduler(task_list, today_duration=today_duration,
                                  with_today=with_today)
    return final_tasks


def generate_overwork_error_message(overwork_minutes):
    return f"You have scheduled {overwork_minutes} additional minutes of " \
           f"work for today. Please change your HOURS_TODAY value in the " \
           f"WorkFlowy tree or reduce the amount of work by removing #today " \
           f"for some of the tasks."


def schedule_tasks_for_today(projects, ordered_tasks, duration_remaining):
    task_dict = task_dict_from_projects(projects)
    
    # Schedule tasks marked for today
    today_tasks = []
    overwork_minutes = 0
    
    for task in ordered_tasks:
        task_id = task.get_task_id()
        task_item = task_dict[task_id]
        task_item["val"] = task.get_reward()
        
        # If the task has not been completed and it is marked for today
        if not task_item["completed"] and task_item["today"]:
    
            # If there is enough time to schedule the task
            if duration_remaining >= task_item["est"]:
                today_tasks += [task_item]
                duration_remaining -= task_item["est"]
            else:
                overwork_minutes += task_item["est"]
        
    overwork_minutes -= duration_remaining
    if overwork_minutes > 0:
        raise Exception(generate_overwork_error_message(overwork_minutes))

    # Schedule other tasks
    for task in ordered_tasks:
        
        # If not time left, don't add additional tasks (without #today)
        if duration_remaining == 0:
            break
            
        task_id = task.get_task_id()
        task_item = task_dict[task_id]
        
        # If the task has not been completed and it is not for today and
        # there is enough time to complete the task today
        if (task_item["est"] <= duration_remaining) \
                and not task_item["completed"] \
                and not task_item["future"] \
                and not task_item["today"]:
            today_tasks += [task_item]
            duration_remaining -= task_item["est"]

    return today_tasks
