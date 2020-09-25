from collections import deque

from pprint import pprint
from src.schedulers.helpers import *
from src.utils import item_dict_from_projects


def basic_scheduler(item_list, current_day, duration_remaining=8 * 60,
                    with_today=True):
    """
    Takes in flattened project tree with "reward" from some API
    Outputs list of items for today
    """

    # Initialize queue of items for today
    today_items = deque()

    # Initialize queue of other items eligible to be scheduled today
    remaining_items = deque()

    # Get information on current weekday
    current_weekday = current_day.weekday()

    if with_today:
        for item in item_list:
    
            # If item is not completed and not indefinitely postponed (future)
            if not item["completed"] and not item["future"]:
        
                # If item is marked to be scheduled today by the user
                if item["scheduled_today"]:
                    today_items.append(item)
                    duration_remaining -= item["est"]
        
                # If item is should be repetitively scheduled on today's day
                elif is_repetitive_item(item, weekday=current_weekday):
                    today_items.append(item)
                    duration_remaining -= item["est"]
        
                # If the item is eligible to be scheduled today
                elif check_additional_scheduling(
                        item, current_day, current_weekday):
                    remaining_items.append(item)
        
    # From: https://stackoverflow.com/a/73050
    sorted_by_deadline = sorted(list(remaining_items),
                                key=lambda k: k['deadline'])
    for item in sorted_by_deadline:
        
        # If no time left, don't add additional items (without #today)
        if duration_remaining == 0:
            break

        # If there is enough time to schedule item
        if item["est"] <= duration_remaining:
            today_items.append(item)
            duration_remaining -= item["est"]

    return list(today_items)


def deadline_scheduler(item_list, current_day, deadline_window=1,
                       today_duration=8 * 60, with_today=True):
    # items within deadline window are tagged with today
    for item in item_list:
        if item["deadline"] <= deadline_window:
            item["today"] = True
    
    final_items = basic_scheduler(item_list, current_day=current_day,
                                  duration_remaining=today_duration,
                                  with_today=with_today)
    return final_items


def schedule_items_for_today(projects, ordered_items, duration_remaining,
                             current_day):
    
    # Get item dictionary from JSON tree
    item_dict = item_dict_from_projects(projects)
    
    # Initialize queue of items for today
    today_items = deque()
    
    # Initialize queue of other items eligible to be scheduled today
    remaining_items = deque()
    
    # Get information on current weekday
    current_weekday = current_day.weekday()

    for item in ordered_items:
        
        item_id = item.get_id()
        item_item = item_dict[item_id]
        
        item_item["est"] = item.get_time_est()
        item_item["val"] = item.get_optimal_reward()

        # If the item is not completed and not indefinitely postponed (future)
        if not item_item["completed"] and not item_item["future"]:
            
            # If item is marked to be scheduled today by the user
            if item_item["scheduled_today"]:
                today_items.append(item_item)
                duration_remaining -= item_item["est"]

            # If item is should be repetitively scheduled on today's day
            elif is_repetitive_item(item_item, weekday=current_weekday):
                today_items.append(item_item)
                duration_remaining -= item_item["est"]

            # If the item is eligible to be scheduled today
            elif check_additional_scheduling(item_item,
                                             current_day, current_weekday):
                remaining_items.append(item_item)
                
    # Schedule other items if time left
    while len(remaining_items) > 0 and duration_remaining > 0:

        # Get next item in the list
        item_item = remaining_items.popleft()
        
        # If there is enough time to schedule item
        if item_item["est"] <= duration_remaining:
            today_items.append(item_item)
            duration_remaining -= item_item["est"]
        
    today_items = list(today_items)
    today_items.sort(key=lambda item: -item["val"])
    
    return today_items
