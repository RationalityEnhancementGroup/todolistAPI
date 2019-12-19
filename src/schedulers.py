def basic_scheduler(task_list, today_duration=8 * 60, with_today=True):
    """
    Takes in flattened project tree with "reward" from some API
    Outputs list of tasks for today
    """
    duration_remaining = today_duration
    
    if not with_today:
        final_tasks = []
    else:
        final_tasks = [task for task in task_list if task["today"] == 1]
        duration_remaining -= sum([task["est"] for task in final_tasks])

    # From: https://stackoverflow.com/a/73050
    sorted_by_value = sorted(task_list, key=lambda k: k['val'])
    for task in sorted_by_value[::-1]:
        if task["est"] <= duration_remaining:
            final_tasks.append(task)
            duration_remaining -= task["est"]
    
    return final_tasks


def deadline_scheduler(task_list, deadline_window=1, today_duration=8 * 60,
                       with_today=True):
    # Tasks within deadline window are tagged with today
    for task in task_list:
        if task["deadline"] <= deadline_window:
            task["today"] = 1
    
    final_tasks = basic_scheduler(task_list, today_duration=today_duration,
                                  with_today=with_today)
    return final_tasks
