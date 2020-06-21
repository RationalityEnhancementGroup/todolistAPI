import json
import os
import stopit
import sys

from copy import deepcopy
from datetime import datetime
from pprint import pprint
from pymongo import MongoClient, DESCENDING

from src.apis import *
from todolistMDP.smdp_test_generator import *
from src.schedulers.schedulers import *
from src.utils import *

# from todolistMDP.mdp_solvers \
#     import backward_induction, policy_iteration, value_iteration

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
                
                points_per_hour = vpath[-5]
                rounding = vpath[-4]
                user_key = vpath[-2]
                api_method = vpath[-1]
                
                # JSON tree parameters
                try:
                    time_zone = int(jsonData["timezoneOffsetMinutes"])
                except:
                    status = "Missing time zone info in JSON object. Please " \
                             "contact us at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                # Set up current time and date according to the user (UTC + TZ)
                user_datetime = datetime.utcnow() + timedelta(minutes=time_zone)
                log_dict["user_datetime"] = user_datetime

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
                
                # Set up current time and date according to the user (UTC + TZ)
                user_datetime = datetime.utcnow() + timedelta(minutes=time_zone)
                log_dict["user_datetime"] = user_datetime

                # Additional parameters (the order of URL input matters!)
                # Casting to other data types is done in the functions that use
                # these parameters
                parameters = [item for item in vpath[9:-5]]

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

                # Is there a user key
                try:
                    log_dict["user_id"] = jsonData["userkey"]
                except:
                    status = "We encountered a problem with the inputs " \
                             "from Workflowy, please try again."
                    store_log(db.request_log, log_dict, status=status)
                    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)

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
                                          default_time_est, user_datetime)

                # Subtract daily tasks time estimation from typical working hours
                for weekday in range(len(typical_minutes)):
                    typical_minutes[weekday] -= daily_tasks_time_est[weekday]
                    
                log_dict["typical_daily_minutes"] = typical_minutes
                
                # Stop timer: parsing scheduling tags
                toc = time.time()

                # Store time: parsing scheduling tags
                timer["Parsing scheduling tags"] = toc - tic

                # Check whether users have assigned more tasks than their time allows.
                # for weekday in range(len(typical_minutes)):
                #     if typical_minutes[weekday] < 0:
                #         # TODO: Change the error message once we introduce weekdays in WorkFlowy
                #         status = f"You have {-typical_minutes[weekday]} more " \
                #                  f"minutes assigned on a typical day. Please " \
                #                  f"increase your typical working hours or " \
                #                  f"remove some of the #daily tasks. "
                #
                #         # Store error in DB
                #         store_log(db.request_log, log_dict, status=status)
                #
                #         cherrypy.response.status = 403
                #         return json.dumps(status + CONTACT)

                # Start timer: subtracting times
                tic = time.time()

                # Subtract time estimation of current intentions from available time
                for tasks in current_intentions.values():
                    for task in tasks:
                        # If the task is not marked as completed or "nevermind"
                        if not task["d"] and not task["nvm"]:
                            today_minutes -= task["est"]

                # Subtract time estimate for all tasks scheduled by the user on
                # today's date from the total number of minutes for today
                for goal in projects:
                    today_minutes -= goal["today_est"]
                
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

                # DP method
                # elif method == "dp" or method == "greedy":
                #     # Get mixing parameter | Default URL value: 0
                #     mixing_parameter = float(parameters[0])
                #
                #     # Store the value of the mixing parameter in the log dict
                #     log_dict['mixing_parameter'] = mixing_parameter
                #
                #     # Defined by the experimenter
                #     if not (0 <= mixing_parameter < 1):
                #         status = "There was an issue with the API input " \
                #                  "(mixing-parameter value). Please contact " \
                #                  "us at reg.experiments@tuebingen.mpg.de."
                #         store_log(db.request_log, log_dict, status=status)
                #         cherrypy.response.status = 403
                #         return json.dumps(status)
                #
                #     scaling_inputs = {
                #         'scale_type': "no_scaling",
                #         'scale_min': None,
                #         'scale_max': None
                #     }
                #
                #     if len(parameters) >= 4:
                #         scaling_inputs['scale_type'] = parameters[1]
                #         scaling_inputs['scale_min'] = float(parameters[2])
                #         scaling_inputs['scale_max'] = float(parameters[3])
                #
                #         if scaling_inputs["scale_min"] == float("inf"):
                #             scaling_inputs["scale_min"] = None
                #         if scaling_inputs["scale_max"] == float("inf"):
                #             scaling_inputs["scale_max"] = None
                #
                #     solver_fn = run_dp_algorithm
                #     if method == "greedy":
                #         solver_fn = run_greedy_algorithm
                #
                #     # TODO: Get informative exceptions
                #     try:
                #         final_tasks = \
                #             assign_dynamic_programming_points(
                #                 projects,
                #                 solver_fn=solver_fn,
                #                 scaling_fn=utility_scaling,
                #                 scaling_inputs=scaling_inputs,
                #                 day_duration=today_minutes,
                #                 mixing_parameter=mixing_parameter,
                #                 time_zone=time_zone,
                #                 verbose=False
                #             )
                #     except Exception as error:
                #         cherrypy.response.status = 403
                #         if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                #
                #             error = "The API took too long processing your " \
                #                     "Workflowy information, please try again. " \
                #                     "Long processing times can arise from too " \
                #                     "many or very late deadlines -- if you " \
                #                     "can, you might want to reduce these."
                #
                #         return json.dumps(str(error) + " " + CONTACT)
                
                elif method == "smdp":
                    
                    # Start timer: reading SMDP parameters
                    tic = time.time()
                    
                    smdp_params = {
                        "choice_mode": parameters[0],
                        "gamma": float(parameters[1]),  # 0.9999
                        "hard_deadline": bool(int(parameters[2])),  # True
                        "loss_rate": - float(parameters[3]),  # -1
                        "num_bins": int(parameters[4]),  # 1
                        "planning_fallacy_const": float(parameters[5]),  # 1.39
                        "slack_reward": float(parameters[6]),  # 1e-3
                        "unit_penalty": float(parameters[7]),  # 1e-1
    
                        "goal_pr_loc": float(parameters[8]),  # 1000
                        "goal_pr_scale": float(parameters[9]),  # 1 - gamma
                        "task_pr_loc": float(parameters[10]),  # 0
                        "task_pr_scale": float(parameters[11]),  # 2
                        
                        'scale_type': "no_scaling",
                        'scale_min':  None,
                        'scale_max':  None
                    }
                    
                    if smdp_params["slack_reward"] == 0:
                        smdp_params["slack_reward"] = np.NINF
    
                    if len(parameters) >= 15:
                        smdp_params['scale_type'] = parameters[12]
                        smdp_params['scale_min'] = float(parameters[13])
                        smdp_params['scale_max'] = float(parameters[14])
        
                        if smdp_params["scale_min"] == float("inf"):
                            smdp_params["scale_min"] = None
                        if smdp_params["scale_max"] == float("inf"):
                            smdp_params["scale_max"] = None

                    # Stop timer: reading SMDP parameters
                    toc = time.time()

                    # Store time: reading SMDP parameters
                    timer["Reading SMDP parameters"] = toc - tic
                    
                    if api_method == "getTasksForToday":
                        final_tasks = assign_smdp_points(
                            projects, timer=timer, day_duration=today_minutes,
                            smdp_params=smdp_params, time_zone=time_zone
                        )

                    else:
                        
                        # Start timer: generating test case
                        tic = time.time()

                        if api_method == "bestSpeedTestSMDP":
                            test_goals = generate_test_case(
                                n_goals=jsonData["n_goals"],
                                n_tasks=jsonData["n_tasks"],
                                worst=False
                            )

                        if api_method == "worstSpeedTestSMDP":
                            test_goals = generate_test_case(
                                n_goals=jsonData["n_goals"],
                                n_tasks=jsonData["n_tasks"],
                                worst=True
                            )

                        # Stop timer: generating test case
                        toc = time.time()
                        
                        # Store time: generating test case
                        timer["Generating test case"] = toc - tic
    
                        # Start timer: Run SMDP
                        tic = time.time()
    
                        final_tasks = assign_smdp_points(
                            projects=test_goals,
                            timer=timer,
                            day_duration=today_minutes,
                            smdp_params=smdp_params,
                            time_zone=time_zone,
                            json=False
                        )
                        
                        # Stop timer: Run SMDP
                        toc = time.time()
                        
                        # Store time: Run SMDP
                        timer["Run SMDP"] = toc - tic

                        # Start timer: Simulating task scheduling
                        tic = time.time()

                        simulate_task_scheduling(test_goals)
                        
                        # Stop timer: Simulating task scheduling
                        toc = time.time()
                        
                        timer["Simulating task scheduling"] = toc - tic
    
                # TODO: Test and fix potential bugs!
                # elif method == "old-report":
                #     final_tasks = \
                #         assign_old_api_points(projects, backward_induction,
                #                               duration=today_minutes)
                
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
                            basic_scheduler(task_list, time_zone=time_zone,
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
                            deadline_scheduler(task_list, time_zone=time_zone,
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
                
                elif api_method == "getTasksForToday":
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

                    # Return scheduled tasks
                    return json.dumps(final_tasks)
                
                elif api_method in ["bestSpeedTestSMDP", "worstSpeedTestSMDP"]:
    
                    # Stop timer: Complete SMDP procedure
                    main_toc = time.time()
    
                    status = f"Testing {api_method} with " \
                             f"{jsonData['n_goals']} goals and " \
                             f"{jsonData['n_tasks']} tasks per goal " \
                             f"took {main_toc - main_tic:.3f} seconds!"
                    
                    # Stop timer: Complete SMDP procedure
                    timer["Complete SMDP procedure"] = main_toc - main_tic

                    return json.dumps({
                        "status": status,
                        "timer": timer
                    })
                
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
    conn = MongoClient(os.environ['MONGODB_URI'] + "?retryWrites=false")
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
