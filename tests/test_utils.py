import pytest

from copy import deepcopy
from datetime import datetime
from src.utils import *

ALL_FALSE = [False for _ in range(7)]
ALL_TRUE = [True for _ in range(7)]

MONDAY = deepcopy(ALL_FALSE)
MONDAY[0] = True

TUESDAY = deepcopy(ALL_FALSE)
TUESDAY[1] = True

WEDNESDAY = deepcopy(ALL_FALSE)
WEDNESDAY[2] = True

THURSDAY = deepcopy(ALL_FALSE)
THURSDAY[3] = True

FRIDAY = deepcopy(ALL_FALSE)
FRIDAY[4] = True

SATURDAY = deepcopy(ALL_FALSE)
SATURDAY[5] = True

SUNDAY = deepcopy(ALL_FALSE)
SUNDAY[6] = True

WEEKDAY = [True for _ in range(5)] + [False for _ in range(2)]
WEEKEND = [False for _ in range(5)] + [True for _ in range(2)]

TODAY = datetime.today().date()
TODAY = f"{TODAY.year}-{TODAY.month}-{TODAY.day}"

TOMORROW = datetime.today().date()
TOMORROW = f"{TOMORROW.year}-{TOMORROW.month}-{TOMORROW.day + 1}"

TREE = [
    {
        "nm": "G1",
        "ch": [
            {
                "nm": "Leaf task G1-T1",
                        "ch": []
            },
            {
                "nm": "Leaf task without ch key G1-T2"
            }
        ]
    },
    {
        "nm": "G2",
        "ch": [
            {
                "nm": "G2-T1",
                "ch": [
                    {
                        "nm": "Leaf task G2-T1-T1",
                        "ch": []
                    },
                    {
                        "nm": "Leaf task G2-T1-T2",
                        "ch": []
                    },
                    {
                        "nm": "G2-T1-T3",
                        "ch": [
                            {
                                "nm": "Leaf task G2-T1-T3-T1",
                                "ch": []
                            }
                        ]
                    }
                ]
            },
            {
                "nm": "G2-T2",
                "ch": [
                    {
                        "nm": "Leaf task without ch key G2-T2"
                    }
                ]
            }
        ]
    }
]


TAGS_TREE = [
    {
        "nm": "G3 Flat goal with repetitive tasks",
        "ch": [
            {
                "nm": "Today task ~~1min #today",
                "ch": []
            },
            {
                "nm": "Daily task ~~1000min #daily",
                "ch": []
            },
            {
                "nm": "Weekday task ~~100min #weekdays",
                "ch": []
            },
            {
                "nm": "Monday task ~~1min #monday",
                "ch": []
            },
            {
                "nm": "Tuesday task ~~1min #tuesday",
                "ch": []
            },
            {
                "nm": "Wednesday task ~~1min #wednesday",
                "ch": []
            },
            {
                "nm": "Thursday task ~~1min #thursday",
                "ch": []
            },
            {
                "nm": "Friday task ~~1min #friday",
                "ch": []
            },
            {
                "nm": "Saturday task ~~1min #saturday",
                "ch": []
            },
            {
                "nm": "Sunday task ~~1min #sunday",
                "ch": []
            },
            {
                "nm": f"Today date task ~~1min #{TODAY}",
                "ch": []
            },
            {
                "nm": f"Tomorrow date task ~~1min #{TOMORROW}",
                "ch": []
            },
            {
                "nm": "Weekend task ~~100min #weekends",
                "ch": []
            },
            {
                "nm": "Each Monday task ~~10min #mondays",
                "ch": []
            },
            {
                "nm": "Each Tuesday task ~~10min #tuesdays",
                "ch": []
            },
            {
                "nm": "Each Wednesday task ~~10min #wednesdays",
                "ch": []
            },
            {
                "nm": "Each Thursday task ~~10min #thursdays",
                "ch": []
            },
            {
                "nm": "Each Friday task ~~10min #fridays",
                "ch": []
            },
            {
                "nm": "Each Saturday task ~~10min #saturdays",
                "ch": []
            },
            {
                "nm": "Each Sunday task ~~10min #sundays",
                "ch": []
            },
            {
                "nm": "Future (not scheduled!) ~~10000min #future",
                "ch": []
            },
            {
                "nm": "Task without tag ~~1min",
                "ch": []
            }
        ]
    }
]


def test_parse_current_intentions():
    """
    Tests parsing current intentions
    """
    
    # Not done, not neverminded, time in minutes
    result = parse_current_intentions_list([
        {
            "d": False,
            "nvm": False,
            "t": "Not done, not neverminded, float points, time in minutes (takes about 120 minutes) $wf:wfid001",
            "vd": 100.0,
        },
        {
            "d": True,
            "nvm": False,
            "t": "Done, not neverminded, integer points, time in hours (takes about 2.5 hours) $wf:wfid002",
            "vd": 100,
        },
        {
            "d": False,
            "nvm": True,
            "t": "Not done, neverminded, 0 points, both hours and minutes (takes about 1 hour and 120 minutes) $wf:wfid003",
            "vd": 0,
        },
        {
            "t": "Done and neverminded not provided, negative points, no time estimate, no WF id",
            "vd": -1,
        },
        {
            "t": "Task without info (d, nvm, est, vd)"
        }
    ])
    key = "wfid001"
    assert key in result.keys()
    task = result[key][0]
    assert task["d"] == False
    assert task["est"] == 120
    assert task["id"] == key
    assert task["nvm"] == False
    assert task["vd"] == 100
    
    key = "wfid002"
    assert key in result.keys()
    task = result[key][0]
    assert task["d"] == True
    assert task["est"] == 150
    assert task["id"] == key
    assert task["nvm"] == False
    assert task["vd"] == 100
    
    key = "wfid003"
    assert key in result.keys()
    task = result[key][0]
    assert task["d"] == False
    assert task["est"] == 180
    assert task["id"] == key
    assert task["nvm"] == True
    assert task["vd"] == 0
    
    key = "__no_wf_id__"
    assert key in result.keys()
    task = result[key][0]
    assert task["d"] == False
    assert task["est"] == 0
    assert task["id"] == key
    assert task["nvm"] == False
    assert task["vd"] < 0
    
    key = "__no_wf_id__"
    assert key in result.keys()
    task = result[key][1]
    assert task["d"] == False
    assert task["est"] == 0
    assert task["id"] == key
    assert task["nvm"] == False
    assert task["vd"] is None
    

def test_flatten_intentions():
    result = flatten_intentions(deepcopy(TREE))
    assert len(result[0]["ch"]) == 2
    assert len(result[1]["ch"]) == 7
    
    
def test_get_leaf_intentions():
    result = get_leaf_intentions(deepcopy(TREE))
    assert len(result[0]["ch"]) == 2
    assert len(result[1]["ch"]) == 4
    for goal in result:
        for task in goal["ch"]:
            assert "Leaf task" in task["nm"]
    assert True


def test_calculate_tasks_time_est():
    """
    Tests:
    - Whether time calculation of repetitive tasks is corret
    """
    
    """
    Test whether the time calculation of repetitive tasks is correct
    - singular days = 1 (should not be taken into account)
    - plural days = 10 minutes
    - weekdays & weekends = 100 minutes
    - daily = 1000 minutes
    - future = 10000 minutes
    """
    result = parse_scheduling_tags(TAGS_TREE, float('inf'), default_time_est=1)
    assert result == [1110 for _ in range(7)]
    
    """
    Test whether all tags are properly assigned
    """
    task = TAGS_TREE[0]["ch"][0]  #today
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is True
    assert task["task_days"] == ALL_FALSE
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][1]  #daily
    assert task["daily"] is True
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == ALL_FALSE  # TODO: ?
    assert task["repetitive_task_days"] == ALL_TRUE

    task = TAGS_TREE[0]["ch"][2]  #weekdays
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == WEEKDAY  # TODO: ?
    assert task["repetitive_task_days"] == WEEKDAY

    task = TAGS_TREE[0]["ch"][3]  #monday
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == MONDAY
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][4]  #tuesday
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == TUESDAY
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][5]  #wednesday
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == WEDNESDAY
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][6]  #thursday
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == THURSDAY
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][7]  #friday
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == FRIDAY
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][8]  #saturday
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == SATURDAY
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][9]  #sunday
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == SUNDAY
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][10]  #TODAY date
    assert task["daily"] is False
    assert task["day_datetime"] == datetime.strptime(TODAY + " 23:59", "%Y-%m-%d %H:%M")
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == ALL_FALSE
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][11]  #TOMORROW date
    assert task["daily"] is False
    assert task["day_datetime"] == datetime.strptime(TOMORROW + " 23:59", "%Y-%m-%d %H:%M")
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == ALL_FALSE
    assert task["repetitive_task_days"] == ALL_FALSE
    
    task = TAGS_TREE[0]["ch"][12]  #weekends
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == [False for _ in range(5)] + [True, True]  # TODO: ?
    assert task["repetitive_task_days"] == [False for _ in range(5)] + [True, True]

    task = TAGS_TREE[0]["ch"][13]  #mondays
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == MONDAY
    assert task["repetitive_task_days"] == MONDAY

    task = TAGS_TREE[0]["ch"][14]  #tuesdays
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == TUESDAY
    assert task["repetitive_task_days"] == TUESDAY

    task = TAGS_TREE[0]["ch"][15]  #wednesdays
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == WEDNESDAY
    assert task["repetitive_task_days"] == WEDNESDAY

    task = TAGS_TREE[0]["ch"][16]  #thursdays
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == THURSDAY
    assert task["repetitive_task_days"] == THURSDAY

    task = TAGS_TREE[0]["ch"][17]  #fridays
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == FRIDAY
    assert task["repetitive_task_days"] == FRIDAY
    
    task = TAGS_TREE[0]["ch"][18]  #saturdays
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == SATURDAY
    assert task["repetitive_task_days"] == SATURDAY

    task = TAGS_TREE[0]["ch"][19]  #sundays
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == SUNDAY
    assert task["repetitive_task_days"] == SUNDAY

    task = TAGS_TREE[0]["ch"][20]  #future
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is True
    assert task["today"] is False
    assert task["task_days"] == ALL_FALSE
    assert task["repetitive_task_days"] == ALL_FALSE

    task = TAGS_TREE[0]["ch"][21]  # without tags
    assert task["daily"] is False
    assert task["day_datetime"] is None
    assert task["future"] is False
    assert task["today"] is False
    assert task["task_days"] == ALL_FALSE
    assert task["repetitive_task_days"] == ALL_FALSE


# TODO: Important functions to test next
#     - test_create_projects_to_save
#     - test_date_str_to_datetime
#     - test_get_final_output
#     - def test_process_deadline
#       - Complex calculations, better to test it manually...
#     - test_create_projects_to_save
#     - test_parse_error_info

# TODO: Other important functions to test
#     - test_compute_latest_start_time (together with tests for the DP method)
#     - test_parse_tree (too complex, better test all functions called by this one)

# TODO: Other functions to test
#     - test_tree_to_old_structure
#     - test_separate_tasks_with_deadlines (not needed after DP speed-ups)
