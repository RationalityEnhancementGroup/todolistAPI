import re
from datetime import datetime

goalCodeRegex = r"#CG(\d|&|_)"
totalValueRegex = r"(?:^| |>)\(?==(\d+)\)?(?:\b|$)"
timeEstimateRegex = r"(?:^| |>)\(?~~(\d+|\.\d+|\d+.\d+)(?:(h(?:ou)?(?:r)?)?(m(?:in)?)?)?s?\)?([^\da-z.]|$)"
deadlineRegex = r"DUE:(\d\d\d\d-\d+-\d+)(?:\b|$)"
todayRegex = r"#today(?:\b|$)"

def flatten_intentions(projects):
    for goal in projects:
        curr_id = goal["id"]
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
            
        for child_idx, child in enumerate(goal["ch"]):
            time_est = re.search(timeEstimateRegex, child["nm"], re.IGNORECASE)
            if time_est[2] is not None: #then this is in hours, convert to minutes
                duration = 60*int(time_est[1])
            else:
                duration = int(time_est[1])
            child["duration"] = duration

            today_code = re.search(todayRegex, child["nm"], re.IGNORECASE)
            if today_code:
                child["today"] = 1
            else:
                child["today"] = 0
    return projects, missing_deadlines, missing_durations
