import json
import os
import stopit
import sys

from pymongo import MongoClient

from src.apis import *
from src.schedulers.schedulers import *
from src.utils import *
from todolistMDP.smdp_test_generator import *

CONTACT = "If you continue to encounter this issue, please contact us at " \
          "reg.experiments@tuebingen.mpg.de."
TIMEOUT_SECONDS = 28

# Extend recursion limit
sys.setrecursionlimit(10000)


class RESTResource(object):
    """
    Base class for providing a RESTful interface to a resource.
    From https://stackoverflow.com/a/2831479
    
    To use this class, simply derive a class from it and implement the methods
    you want to support.  The list of possible methods are:
    handle_GET
    handle_PUT
    handle_POST
    handle_DELETE
    """
    
    @cherrypy.expose
    @cherrypy.tools.accept(media='application/json')
    def default(self, *vpath, **params):
        method = getattr(self, "handle_" + cherrypy.request.method, None)
        if not method:
            methods = [x.replace("handle_", "") for x in dir(self)
                       if x.startswith("handle_")]
            cherrypy.response.headers["Allow"] = ",".join(methods)
            cherrypy.response.status = 405
            status = "Method not implemented."
            return json.dumps(status)
        
        # Can we load the request body (json)
        try:
            rawData = cherrypy.request.body.read()
            jsonData = json.loads(rawData)
        except:
            cherrypy.response.status = 403
            status = "No request body"
            return json.dumps(status)

        return method(jsonData, *vpath, **params)


class PostResource(RESTResource):
    
    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        
        # Start timer
        main_tic = time.time()

        with stopit.ThreadingTimeout(TIMEOUT_SECONDS) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING

            # Initialize log dictionary
            log_dict = {
                "start_time": datetime.now(),
            }
            
            # Initialize dictionary that inspects time taken by each method
            timer = dict()

            try:
                
                api_method = vpath[-1]
                
                # Load fixed time if provided
                if "time" in jsonData.keys():
                    user_datetime = datetime.strptime(jsonData["time"],
                                                      "%Y-%m-%d %H:%M")
                else:
                    user_datetime = datetime.utcnow()

                if api_method in {"averageSpeedTestSMDP",
                                  "bestSpeedTestSMDP",
                                  "exhaustiveSpeedTestSMDP",
                                  "realSpeedTestSMDP",
                                  "worstSpeedTestSMDP"}:
        
                    smdp_params = {
                        "choice_mode":            jsonData["choice_mode"],
                        "gamma":                  jsonData["gamma"],
                        "loss_rate":              jsonData["loss_rate"],
                        "num_bins":               jsonData["n_bins"],
                        "planning_fallacy_const": jsonData["planning_fallacy_const"],
                        "slack_reward":           jsonData["slack_reward"],
                        "unit_penalty":           jsonData["unit_penalty"],
            
                        "scale_type": jsonData["scale_type"],
                        "scale_min":  jsonData["scale_min"],
                        "scale_max":  jsonData["scale_max"],
    
                        "bias": None,
                        "scale": None
                    }

                    if smdp_params["slack_reward"] == 0:
                        smdp_params["slack_reward"] = np.NINF
        
                    """ Generating test case """
                    tic = time.time()
        
                    test_goals = generate_test_case(
                        api_method=api_method,
                        n_bins=jsonData["n_bins"],
                        n_goals=jsonData["n_goals"],
                        n_tasks=jsonData["n_tasks"],
                        time_est=jsonData["time_est"],
                        unit_penalty=jsonData["unit_penalty"]
                    )
                    
                    toc = time.time()
                    timer["Generating test case"] = toc - tic
        
                    """ Run SMDP """
                    tic = time.time()
        
                    assign_smdp_points(
                        projects=test_goals,
                        current_day=user_datetime,
                        day_duration=jsonData["today_minutes"],
                        smdp_params=smdp_params,
                        timer=timer,
                        json=False
                    )
        
                    toc = time.time()
                    timer["Run SMDP"] = toc - tic
        
                    """ Simulating task scheduling """
                    tic = time.time()
        
                    simulate_task_scheduling(test_goals)
        
                    toc = time.time()
                    timer["Simulating task scheduling"] = toc - tic
        
                    # Stop timer: Complete SMDP procedure
                    main_toc = time.time()
        
                    status = f"Testing took {main_toc - main_tic:.3f} seconds!"
        
                    # Stop timer: Complete SMDP procedure
                    timer["Complete SMDP procedure"] = main_toc - main_tic
        
                    return json.dumps({
                        "status": status,
                        "timer":  timer
                    })
    
                # Start timer: reading parameters
                tic = time.time()
                
                # Compulsory parameters
                method = vpath[0]
                scheduler = vpath[1]
                default_time_est = vpath[2]  # This has to be in minutes
                default_deadline = vpath[3]  # This has to be in days
                allowed_task_time = vpath[4]
                min_sum_of_goal_values = vpath[5]
                max_sum_of_goal_values = vpath[6]
                min_goal_value_per_goal_duration = vpath[7]
                max_goal_value_per_goal_duration = vpath[8]
                points_per_hour = vpath[9]
                rounding = vpath[10]
                
                # tree = vpath[-3]  # Dummy parameter
                user_key = vpath[-2]
                api_method = vpath[-1]
                
                # Additional parameters (the order of URL input matters!)
                # Casting to other data types is done in the functions that use
                # these parameters
                parameters = [item for item in vpath[11:-3]]

                # Is there a user key
                try:
                    log_dict["user_id"] = jsonData["userkey"]
                except:
                    status = "We encountered a problem with the inputs " \
                             "from Workflowy, please try again."
                    store_log(db.request_log, log_dict, status=status)
    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)

                # Initialize SMDP parameters
                smdp_params = dict({
                    "planning_fallacy_const": 1
                })
                
                """ Reading SMDP parameters """
                if method == "smdp":
                    
                    try:
                        tic = time.time()
        
                        smdp_params.update({
                            "choice_mode":            parameters[0],
                            "gamma":                  float(parameters[1]),
                            "loss_rate":              - float(parameters[2]),
                            "num_bins":               int(parameters[3]),
                            "planning_fallacy_const": float(parameters[4]),
                            "slack_reward":           float(parameters[5]),
                            "unit_penalty":           float(parameters[6]),
            
                            'scale_type':             "no_scaling",
                            'scale_min':              None,
                            'scale_max':              None
                        })
        
                        if smdp_params["slack_reward"] == 0:
                            smdp_params["slack_reward"] = np.NINF
        
                        if len(parameters) >= 10:
                            smdp_params['scale_type'] = parameters[7]
                            smdp_params['scale_min'] = float(parameters[8])
                            smdp_params['scale_max'] = float(parameters[9])
            
                            if smdp_params["scale_min"] == float("inf"):
                                smdp_params["scale_min"] = None
                            if smdp_params["scale_max"] == float("inf"):
                                smdp_params["scale_max"] = None

                        smdp_params["bias"] = None
                        smdp_params["scale"] = None

                        # Get bias and scale values from database (if available)
                        query = list(db.pr_transform.find(
                            {
                                "user_id": jsonData["userkey"]
                            }
                        ))
                        if len(query) > 0:
                            query = query[-1]
                            smdp_params["bias"] = query["bias"]
                            smdp_params["scale"] = query["scale"]
                        else:
                            query = None
                            
                        if "bias" in jsonData.keys():
                            smdp_params["bias"] = jsonData["bias"]
        
                        if "scale" in jsonData.keys():
                            smdp_params["scale"] = jsonData["scale"]
        
                        toc = time.time()
                        timer["Reading SMDP parameters"] = toc - tic
                        
                    except:
                        status = "There was an issue with the API input " \
                                 "(reading SMDP parameters) Please contact " \
                                 "us at reg.experiments@tuebingen.mpg.de."
                        store_log(db.request_log, log_dict, status=status)
                        cherrypy.response.status = 403
                        return json.dumps(status)

                # JSON tree parameters
                try:
                    time_zone = int(jsonData["timezoneOffsetMinutes"])
                except:
                    status = "Missing time zone info in JSON object. Please " \
                             "contact us at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                # Last two input parameters
                try:
                    round_param = int(rounding)
                except:
                    status = "There was an issue with the API input " \
                             "(rounding parameter). Please contact us at " \
                             "reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                try:
                    points_per_hour = (points_per_hour.lower() in ["true", "1", "t", "yes"])
                except:
                    status = "There was an issue with the API input (point " \
                             "per hour vs completion parameter). Please " \
                             "contact us at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                # Update time with time zone
                user_datetime += timedelta(minutes=time_zone)
                log_dict["user_datetime"] = user_datetime

                log_dict.update({
                    "api_method": api_method,
                    "default_time_est": default_time_est,
                    "default_deadline": default_deadline,
                    "allowed_task_time": allowed_task_time,
                    "duration": str(datetime.now() - log_dict["start_time"]),
                    "method": method,
                    "parameters": parameters,
                    "round_param": round_param,
                    "points_per_hour": points_per_hour,
                    "scheduler": scheduler,
                    "time_zone": time_zone,
                    "user_key": user_key,
                    
                    "min_sum_of_goal_values": min_sum_of_goal_values,
                    "max_sum_of_goal_values": max_sum_of_goal_values,
                    "min_goal_value_per_goal_duration":
                        min_goal_value_per_goal_duration,
                    "max_goal_value_per_goal_duration":
                        max_goal_value_per_goal_duration,
                    
                    # Must be provided on each store (if needed)
                    "lm": None,
                    "mixing_parameter": None,
                    "status": None,
                    "timestamp": None,
                    "user_id": None,
                })
                
                # Parse default time estimation (in minutes)
                try:
                    default_time_est = int(default_time_est)
                    log_dict["default_time_est"] = default_time_est
                except:
                    status = "There was an issue with the API input " \
                             "(default time estimation). Please contact us " \
                             "at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)

                # Parse default deadline (in days)
                try:
                    default_deadline = int(default_deadline)
                    log_dict["default_deadline"] = default_deadline
                except:
                    status = "There was an issue with the API input " \
                             "(default deadline). Please contact us at " \
                             "reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)

                # Get allowed task time | Default URL value: 'inf'
                try:
                    allowed_task_time = float(allowed_task_time)
                    log_dict["allowed_task_time"] = allowed_task_time
                except:
                    status = "There was an issue with the API input " \
                             "(allowed time parameter). Please contact us at " \
                             "reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                try:
                    min_sum_of_goal_values = float(min_sum_of_goal_values)
                    max_sum_of_goal_values = float(max_sum_of_goal_values)
                    min_goal_value_per_goal_duration = \
                        float(min_goal_value_per_goal_duration)
                    max_goal_value_per_goal_duration = \
                        float(max_goal_value_per_goal_duration)
                    
                    log_dict["min_sum_of_goal_values"] = min_sum_of_goal_values
                    log_dict["max_sum_of_goal_values"] = max_sum_of_goal_values
                    log_dict["min_goal_value_per_goal_duration"] = \
                        min_goal_value_per_goal_duration,
                    log_dict["max_goal_value_per_goal_duration"] = \
                        max_goal_value_per_goal_duration,
                except:
                    status = "There was an issue with the API input " \
                             "(goal-value limits). Please contact us at " \
                             "reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)

                # Parse today hours
                try:
                    today_hours = parse_hours(jsonData["today_hours"][0]["nm"])
                    log_dict["today_hours"] = today_hours
                except:
                    status = "Something is wrong with the hours in " \
                             "HOURS_TODAY. Please take a look and try again."
                    store_log(db.request_log, log_dict, status=status)
    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)

                # Parse typical hours
                try:
                    typical_hours = parse_hours(
                        jsonData["typical_hours"][0]["nm"])
                    log_dict["typical_hours"] = typical_hours
                except:
                    status = "Something is wrong with the hours in " \
                             "HOURS_TYPICAL. Please take a look and try again."
                    store_log(db.request_log, log_dict, status=status)
    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)

                # Check whether typical hours is in the pre-defined range
                if not (0 < typical_hours <= 24):
                    store_log(db.request_log, log_dict,
                              status="Invalid typical hours value.")
    
                    status = "Please edit the hours you typically work on Workflowy. " \
                             "The hours you work should be between 0 and 24."
                    cherrypy.response.status = 403
                    return json.dumps(status)

                # Check whether today hours is in the pre-defined range
                # 0 is an allowed value in case users want to skip a day
                if not (0 <= today_hours <= 24):
                    store_log(db.request_log, log_dict,
                              status="Invalid today hours value.")
    
                    status = "Please edit the hours you can work today on Workflowy. " \
                             "The hours you work should be between 0 and 24."
                    cherrypy.response.status = 403
                    return json.dumps(status)

                # Convert today hours into minutes
                today_minutes = today_hours * 60

                # Convert typical hours into typical minutes for each weekday
                typical_minutes = [typical_hours * 60 for _ in range(7)]

                # Update last modified
                log_dict["lm"] = jsonData["updated"]
                
                # Stop timer: reading parameters
                toc = time.time()
                
                # Store time: reading parameters
                timer["Reading parameters"] = toc - tic
                
                # Start timer: parsing current intentions
                tic = time.time()
                
                # Parse current intentions
                try:
                    current_intentions = parse_current_intentions_list(
                        jsonData["currentIntentionsList"],
                        default_time_est=default_time_est)
                except:
                    status = "An error related to the current " \
                             "intentions has occurred."
                    
                    # Save the data if there was a change, removing nm fields so
                    # that we keep participant data anonymous
                    store_log(db.request_log, log_dict, status=status)

                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)
                
                # Store current intentions
                log_dict["current_intentions"] = current_intentions

                # Stop timer: parsing current intentions
                toc = time.time()

                # Store time: parsing current intentions
                timer["Parsing current intentions"] = toc - tic
                
                # Start timer: parsing hierarchical structure
                tic = time.time()

                # New calculation + Save updated, user id, and skeleton
                try:
                    flatten_projects = \
                        flatten_intentions(deepcopy(jsonData["projects"]))
                    log_dict["flatten_tree"] = \
                        create_projects_to_save(flatten_projects)
                    
                    projects = get_leaf_intentions(jsonData["projects"])
                    log_dict["tree"] = \
                        create_projects_to_save(projects)
                except:
                    status = "Something is wrong with your inputted " \
                             "goals and tasks. Please take a look at your " \
                             "Workflowy inputs and then try again."
                    
                    # Save the data if there was a change, removing nm fields so
                    # that we keep participant data anonymous
                    store_log(db.request_log, log_dict, status=status)
                    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)
                
                # Stop timer: parsing hierarchical structure
                toc = time.time()
                
                # Store time: parsing hierarchical structure
                timer["Parsing hierarchical structure"] = toc - tic
                
                # Start timer: parsing scheduling tags
                tic = time.time()
                
                # Get information about daily tasks time estimation
                daily_tasks_time_est = \
                    parse_scheduling_tags(projects, allowed_task_time,
                                          default_time_est,
                                          smdp_params["planning_fallacy_const"],
                                          user_datetime)

                # Subtract daily tasks time estimation from typical working hours
                for weekday in range(len(typical_minutes)):
                    typical_minutes[weekday] -= daily_tasks_time_est[weekday]
                    
                log_dict["typical_daily_minutes"] = typical_minutes
                
                # Stop timer: parsing scheduling tags
                toc = time.time()

                # Store time: parsing scheduling tags
                timer["Parsing scheduling tags"] = toc - tic

                # Start timer: subtracting times
                tic = time.time()

                # Subtract time estimation of current intentions from available time
                for tasks in current_intentions.values():
                    for task in tasks:
                        # If the task is not marked as completed or "nevermind"
                        if not task["d"] and not task["nvm"]:
                            today_minutes -= task["est"]

                # Make 0 if the number of minutes is negative
                today_minutes = max(today_minutes, 0)

                # Stop timer: subtracting times
                toc = time.time()

                # Store time: subtracting times
                timer["Subtracting times"] = toc - tic

                # Start timer: parsing to-do list
                tic = time.time()

                # Parse to-do list
                try:
                    projects = \
                        parse_tree(projects, current_intentions,
                                   today_minutes, typical_minutes,
                                   default_deadline=default_deadline,
                                   min_sum_of_goal_values=min_sum_of_goal_values,
                                   max_sum_of_goal_values=max_sum_of_goal_values,
                                   min_goal_value_per_goal_duration=min_goal_value_per_goal_duration,
                                   max_goal_value_per_goal_duration=max_goal_value_per_goal_duration,
                                   user_datetime=user_datetime)
                except Exception as error:
                    status = str(error)
                    
                    # Remove personal data
                    anonymous_error = parse_error_info(status)
                    
                    # Store error in DB
                    store_log(db.request_log, log_dict, status=anonymous_error)
                    
                    status += " Please take a look at your Workflowy inputs " \
                              "and then try again. "
                    cherrypy.response.status = 403
                    return json.dumps(status + CONTACT)
                
                log_dict["tree"] = create_projects_to_save(projects)

                # Stop timer: parsing to-do list
                toc = time.time()

                # Store time: parsing to-do list
                timer["Parsing to-do list"] = toc - tic

                # Start timer: storing parsed to-do list in database
                tic = time.time()

                # Save the data if there was a change, removing nm fields so
                # that we keep participant data anonymous
                store_log(db.request_log, log_dict, status="Save parsed tree")
                
                # Stop timer: storing parsed to-do list in database
                toc = time.time()
                
                # Stop timer: storing parsed to-do list in database
                timer["Storing parsed to-do list in database"] = toc - tic
                
                if method == "constant":
                    # Parse default task value
                    try:
                        default_task_value = float(parameters[0])
                    except:
                        status = "Error while parsing default task value. "
                        
                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)
    
                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)
                    
                    # Assign constant points
                    try:
                        projects = assign_constant_points(projects,
                                                          default_task_value)
                    except:
                        status = "Problem while assigning points. "
                        
                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)

                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)
                    
                elif method == "length":
                    # Assign random points
                    try:
                        projects = assign_length_points(projects)
                    except:
                        status = "Problem while assigning points. "
        
                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)
        
                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)

                elif method == "random":
                    # Parse distribution name
                    try:
                        distribution = parameters[0].lower()
                    except:
                        status = "Error while parsing distribution name. "
    
                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)
    
                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)

                    # Parse distribution parameters
                    try:
                        distribution_params = [float(param)
                                               for param in parameters[1:]]
                    except:
                        status = "Error while parsing distribution parameters. "
    
                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)
    
                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)

                    if distribution == 'uniform':
                        distribution = np.random.uniform
                    if distribution == 'normal':
                        distribution = np.random.normal
                        
                    # Assign random points
                    try:
                        projects = assign_random_points(
                            projects, distribution_fxn=distribution,
                            fxn_args=distribution_params)
                    except:
                        status = "Problem while assigning points. "
    
                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)

                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)
                
                elif method == "smdp":
    
                    final_tasks = assign_smdp_points(
                        projects, current_day=user_datetime, timer=timer,
                        day_duration=today_minutes, smdp_params=smdp_params
                    )

                    # Add database entry if one does not exist
                    if query is None:
                        db.pr_transform.insert_one({
                            "user_id": jsonData["userkey"],
                            "bias":    smdp_params["bias"],
                            "scale":   smdp_params["scale"]
                        })

                    if api_method == "updateTransform":
                        
                        # Update bias and scaling parameters
                        if query is not None:
                            db.pr_transform.update_one(
                                {"_id": query["_id"]},
                                {
                                    "$set": {
                                        "bias": smdp_params["bias"],
                                        "scale": smdp_params["scale"]
                                    }
                                }
                            )

                else:
                    status = "API method does not exist. Please contact us " \
                             "at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                # Start timer: Anonymizing data
                tic = time.time()
                
                # Update values in the tree
                log_dict["tree"] = create_projects_to_save(projects)

                # Stop timer: Anonymizing data
                toc = time.time()
                
                # Store time: Anonymizing date
                timer["Anonymize data"] = toc - tic

                # Schedule tasks for today
                if scheduler == "basic":
                    try:
                        # Get task list from the tree
                        task_list = task_list_from_projects(projects)
    
                        final_tasks = \
                            basic_scheduler(task_list,
                                            current_day=user_datetime,
                                            duration_remaining=today_minutes)
                    except Exception as error:
                        status = str(error) + ' '
    
                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)
    
                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)

                elif scheduler == "deadline":
                    try:
                        # Get task list from the tree
                        task_list = task_list_from_projects(projects)
    
                        final_tasks = \
                            deadline_scheduler(task_list,
                                               current_day=user_datetime,
                                               today_duration=today_minutes)
                    except Exception as error:
                        status = str(error) + ' '

                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)

                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)

                elif scheduler == "mdp":
                    pass
                else:
                    status = "Scheduling method does not exist. Please " \
                             "contact us at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                # Start timer: Storing incentivized tree in database
                tic = time.time()
                
                store_log(db.trees, log_dict, status="Save tree!")
                
                # Stop timer: Storing incentivized tree in database
                toc = time.time()
                
                # Store time: Storing incentivized tree in database
                timer["Storing incentivized tree in database"] = toc - tic

                if api_method == "updateTree":
                    cherrypy.response.status = 204
                    store_log(db.request_log, log_dict, status="Update tree")
                    return None
                
                elif api_method in {"getTasksForToday", "updateTransform"}:
                    if len(final_tasks) == 0:
                        status = "The API has scheduled all of the tasks it " \
                                 "can for today, given your working hours. " \
                                 "If you want to pull a specific task, please " \
                                 "tag it #today on Workflowy. You may also " \
                                 "change your working hours for today at the " \
                                 "bottom of the Workflowy tree."
                        store_log(db.request_log, log_dict, status=status)
                        cherrypy.response.status = 403
                        return json.dumps(status + " " + CONTACT)
                    try:
                        
                        # Start timer: Storing human-readable output
                        tic = time.time()
                        
                        final_tasks = get_final_output(
                            final_tasks, round_param, points_per_hour,
                            user_datetime=user_datetime)
                        
                        # Stop timer: Storing human-readable output
                        toc = time.time()
                        
                        # Store time: Storing human-readable output
                        timer["Storing human-readable output"] = toc - tic
                        
                    except NameError as error:
                        store_log(db.request_log, log_dict,
                                  status="Task has no name!")
                        cherrypy.response.status = 403
                        return json.dumps(str(error) + " " + CONTACT)
                    
                    except:
                        status = "Error while preparing final output."
                        store_log(db.request_log, log_dict, status=status)
                        cherrypy.response.status = 403
                        return json.dumps(status + " " + CONTACT)

                    # Start timer: Storing successful pull in database
                    tic = time.time()

                    store_log(db.request_log, log_dict, status="Successful pull!")

                    # Stop timer: Storing successful pull in database
                    toc = time.time()
                    
                    # Store time: Storing successful pull in database
                    timer["Storing successful pull in database"] = toc - tic
                    
                    # print("\n===== Optimal tasks =====")
                    # for task in final_tasks:
                    #     print(task["nm"], "&", task["val"], "\\\\")
                    # print()

                    # Return scheduled tasks
                    return json.dumps(final_tasks)
                
                else:
                    status = "API Method not implemented. Please contact us " \
                             "at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    
                    cherrypy.response.status = 405
                    return json.dumps(status)

            except stopit.utils.TimeoutException:
                return json.dumps("Timeout!")
                
            except Exception as error:
                status = "The API has encountered an error, please try again."
                
                # status = str(error)
                
                # Store anonymous error info in DB collection
                anonymous_error = parse_error_info(str(error))
                try:
                    store_log(db.request_log, log_dict,
                              status=status + " " + anonymous_error)
                except:
                    store_log(db.request_log,
                              {"start_time": log_dict["start_time"],
                               "status": status + " " +
                                         "Exception while storing anonymous error info."})
                
                cherrypy.response.status = 403
                return json.dumps(status + " " + CONTACT)


class Root(object):
    api = PostResource()
    
    @cherrypy.expose
    def index(self):
        return "Server is up!"
        # return "REST API for Complice Project w/ Workflowy Points"


if __name__ == '__main__':
    conn = MongoClient(os.environ['DB_URI'] + "?retryWrites=false")
    db = conn.heroku_g6l4lr9d
    
    conf = {
        '/':       {
            # 'tools.sessions.on': True,
            'tools.response_headers.on':      True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]},
        '/static': {
            'tools.staticdir.on':    True,
            'tools.staticdir.dir':   os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'static'),
            'tools.staticdir.index': 'urlgenerator.html'
        }
    }
    
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update(
        {'server.socket_port': int(os.environ.get('PORT', '6789'))})
    cherrypy.quickstart(Root(), '/', conf)
