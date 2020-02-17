from src.utils import task_dict_from_projects


def basic_scheduler(task_list, today_duration=8 * 60, with_today=True):
    """
    Takes in flattened project tree with "reward" from some API
    Outputs list of tasks for today
    """
    duration_remaining = today_duration
    
    if not with_today:
        final_tasks = []
    else:
        final_tasks = [task for task in task_list
                       if task["today"] and not task["completed"]]
        duration_remaining -= sum([task["est"] for task in final_tasks])

    # From: https://stackoverflow.com/a/73050
    sorted_by_deadline = sorted(task_list, key=lambda k: k['deadline'])
    for task in sorted_by_deadline:
        if (task["est"] <= duration_remaining) \
                and (not task["today"] and not task["completed"]):
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


def schedule_tasks_for_today(projects, ordered_tasks, day_duration):
    task_dict = task_dict_from_projects(projects)
    
    # Schedule tasks marked for today
    today_tasks = []
    for task in ordered_tasks:
        task_id = task.get_task_id()
        task_from_project = task_dict[task_id]
        task_from_project["val"] = task.get_reward()
        
        if task_from_project["today"] and \
                day_duration >= task_from_project["est"]:
            today_tasks += [task_from_project]
            day_duration -= task_from_project["est"]


    # Schedule other tasks
    for task in ordered_tasks:
        task_id = task.get_task_id()
        task_from_project = task_dict[task_id]
        
        # If the task is not marked for today and
        # if there is enough time to complete the task today
        if task_from_project["today"] != 1 and \
                day_duration >= task_from_project["est"]:
            today_tasks += [task_from_project]
            day_duration -= task_from_project["est"]

    return today_tasks
