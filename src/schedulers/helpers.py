def check_additional_scheduling(task, day, weekday):
    return check_date_assignment(task, day) is None and \
           check_weekday_assignment(task, weekday) is None


def check_date_assignment(task, day):
    if task['day_datetime'] is None:
        return None
    
    return task['day_datetime'].date() == day.date()


def check_priority_scheduling(task, day, weekday):
    return (check_date_assignment(task, day) or
            check_weekday_assignment(task, weekday))


def check_weekday_assignment(task, weekday):
    # Check whether there is a weekday preference
    if task["days"][weekday]:
        return True
    
    # Check whether there are weekday preferences
    # - None: There are no weekday preferences
    # - False: There are weekday preferences, but not for today
    weekday_assignment = None
    for day_idx in range(len(task["days"])):
        if task["days"][day_idx] or task["repetitive_days"][day_idx]:
            weekday_assignment = False
            
    # Return whether there are weekday preferences
    return weekday_assignment


def generate_overwork_error_message(overwork_minutes):
    return f"You have scheduled {overwork_minutes} additional minutes of " \
           f"work for today. Please change your HOURS_TODAY value in the " \
           f"WorkFlowy tree or reduce the amount of work by removing #daily " \
           f"or #today for some of the tasks."


def is_repetitive_task(task, weekday):
    return task["daily"] or task["repetitive_days"][weekday]
