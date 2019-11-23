import re
from datetime import datetime

goalCodeRegex = r"#CG(\d|&|_)"
totalValueRegex = r"(?:^| |>)\(?==(\d+)\)?(?:\b|$)"
timeEstimateRegex = r"(?:^| |>)\(?~~(\d+|\.\d+|\d+.\d+)(?:(h(?:ou)?(?:r)?)?(m(?:in)?)?)?s?\)?([^\da-z.]|$)"
deadlineRegex = r"DUE:(\d\d\d\d-\d+-\d+)(?:\b|$)"
todayRegex = r"#today(?:\b|$)"

def parse_hours(time_string, default_hours = 8):
    try:
        return int(re.search(totalValueRegex, time_string, re.IGNORECASE)[1])
    except:
        return default_hours

def flatten_intentions(projects):
    for goal in projects:
        for child in goal["ch"]:
            if "ch" in child:
                goal["ch"].extend(child["ch"])
                del child["ch"]
    return projects

def parse_tree(projects):
    '''
    This function reads in a flattened project tree and parses fields like goal code, total value, duration and deadline
    '''
    missing_deadlines = 0 #we expect this to be 1 since misc is not initialized with a deadline
    missing_durations = 0

    current_date = datetime.now().date()

    for goal in projects:
        #extract goal information
        goalCode = re.search(goalCodeRegex, goal["nm"], re.IGNORECASE)[1]
        value =  int(re.search(totalValueRegex, goal["nm"], re.IGNORECASE)[1])

        try:
            parsedDeadline = re.search(deadlineRegex, goal["nm"], re.IGNORECASE)[1]
            goalDeadline = (datetime.strptime(parsedDeadline, "%Y-%m-%d").date()-current_date).days
        except:
            goalDeadline = None
            missing_deadlines += 1

        goal["deadline"] = goalDeadline
        goal["value"] = value 
        goal["code"] = goalCode

        goal_id = goal["id"]
            
        for child_idx, child in enumerate(goal["ch"]):
            time_est = re.search(timeEstimateRegex, child["nm"], re.IGNORECASE)
            if time_est[2] is not None: #then this is in hours, convert to minutes
                duration = 60*int(time_est[1])
            else:
                duration = int(time_est[1])
            child["est"] = duration

            today_code = re.search(todayRegex, child["nm"], re.IGNORECASE)
            if today_code:
                child["today"] = 1
            else:
                child["today"] = 0

            child["deadline"] = goalDeadline
            child["parentId"] = goal_id
            child["pcp"] = False #TODO not sure what this field is

    return projects, missing_deadlines, missing_durations


def clean_output(task_list):
    '''
    Input is list of tasks
    Outputs list of tasks for today with fields id, nm, lm, parentId, pcp, est, val (=reward)
    '''
    keys_needed = ["id", "nm", "lm", "parentId", "pcp", "est", "val"]

    #for now only look at first dictionary
    current_keys = set(task_list[0].keys())
    extra_keys = list(current_keys-set(keys_needed))
    missing_keys = list(set(keys_needed)-current_keys)

    for extra_key in extra_keys:
        for task in task_list:
            if extra_key in task:
                del task[extra_key]

    for missing_key in missing_keys:
        for task in task_list:
            if missing_key not in task:
                task[missing_key] = None

    for task in task_list:
        task["val"] = round(task["val"])

    return task_list

def task_list_from_projects(projects):
    task_list = []
    for goal in projects:
        task_list.extend(goal["ch"])
    return task_list

def are_there_tree_differences(old_tree, new_tree):
    '''
    input: two trees
    output: whether or not we need to rerun the point calculations (e.g. we don't need to if only day durations change or #today has been added)
    '''
    raise NotImplementedError

def tree_to_old_api_json(projects):
    '''
    input: parsed tree
    output: json that can be inputted to old api
    '''
    old_json = []
    for goal in projects:
        if goal["code"] in [str(goal_num) for goal_num in range(11)]: #don't include miscellaneous
            old_json.append({"title":goal["id"], "tasks":[], "tasks_dur":[], "rewards":{goal["deadline"]:goal["value"]}, "penalty": 0})
            for task in goal["ch"]:
                old_json[-1]["tasks"].append(task["id"])
                old_json[-1]["tasks_dur"].append(task["est"])

    return old_json