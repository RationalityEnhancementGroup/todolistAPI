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
                
                # Load fixed time if provided
                if "time" in jsonData.keys():
                    user_datetime = datetime.strptime(jsonData["time"],
                                                      "%Y-%m-%d %H:%M")
                else:
                    user_datetime = datetime.utcnow()

                # Start timer: reading parameters
                tic = time.time()
                
                # Compulsory parameters
                method = vpath[0]
                scheduler = vpath[1]
                default_time_est = vpath[2]  # ... in minutes
                default_deadline = vpath[3]  # ... in days
                allowed_task_time = vpath[4]  # ... in minutes
                min_sum_of_goal_values = vpath[5]
                max_sum_of_goal_values = vpath[6]
                min_goal_value_per_duration = vpath[7]
                max_goal_value_per_duration = vpath[8]
                points_per_hour = vpath[9]
                rounding = vpath[10]  # Number of decimals
                
                # tree = vpath[-3]  # Dummy parameter
                user_key = vpath[-2]
                api_method = vpath[-1]
                
                # Additional parameters (the order of URL input matters!)
                # Casting to other data types is done within the functions that
                # use these parameters
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
                            "penalty_rate":           float(parameters[6]),
                            # "sub_goal_max_time":      float(parameters[7]),
            
                            'scale_type':             None,
                            'scale_min':              None,
                            'scale_max':              None
                        })
                        
                        if smdp_params["slack_reward"] == 0:
                            smdp_params["slack_reward"] = np.NINF
        
                        if len(parameters) > 7:
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
                        if len(query) > 0 and api_method != "updateTransform":
                            query = query[-1]
                            smdp_params["bias"] = query["bias"]
                            smdp_params["scale"] = query["scale"]
                        else:
                            query = None
                            
                        if "bias" in jsonData.keys():
                            smdp_params["bias"] = jsonData["bias"]
        
                        if "scale" in jsonData.keys():
                            smdp_params["scale"] = jsonData["scale"]
                            
                        if "timeFrame" in jsonData.keys():
                            smdp_params["sub_goal_max_time"] = float(jsonData["timeFrame"])
                        else:
                            smdp_params["sub_goal_max_time"] = 0
        
                        timer["Reading SMDP parameters"] = time.time() - tic
                        
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
                    points_per_hour = (
                        points_per_hour.lower() in ["true", "1", "t", "yes"]
                    )
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
                    "min_goal_value_per_duration":
                        min_goal_value_per_duration,
                    "max_goal_value_per_duration":
                        max_goal_value_per_duration,
                    
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
                    min_goal_value_per_duration = \
                        float(min_goal_value_per_duration)
                    max_goal_value_per_duration = \
                        float(max_goal_value_per_duration)
                    
                    log_dict["min_sum_of_goal_values"] = min_sum_of_goal_values
                    log_dict["max_sum_of_goal_values"] = max_sum_of_goal_values
                    log_dict["min_goal_value_per_duration"] = \
                        min_goal_value_per_duration,
                    log_dict["max_goal_value_per_duration"] = \
                        max_goal_value_per_duration,
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
                if not (0 <= today_hours <= np.PINF):
                    store_log(db.request_log, log_dict,
                              status="Invalid today hours value.")
    
                    status = "Please edit the hours you can work today on Workflowy. " \
                             "The hours you work should be between 0 and 24."
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                # Initialize vector of available time (0: today; 1-7: Mon-Sun)
                available_time =\
                    [today_hours * 60] + [typical_hours * 60 for _ in range(7)]

                # Update last modified
                log_dict["lm"] = jsonData["updated"]
                
                # Store time: reading parameters
                timer["Reading parameters"] = time.time() - tic
                
                # Start timer: parsing current intentions
                tic = time.ti
                
                # Parse current intentions
                try:
                    current_intentions = parse_current_intentions_list(
                        jsonData["currentIntentionsList"])
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

                # Store time: parsing current intentions
                timer["Parsing current intentions"] = time.time() - tic

                # Start timer: parsing generating to-do list
                tic = time.time()

                # Generate to-do list
                try:
                    projects = generate_to_do_list(
                        deepcopy(jsonData["projects"]),
                        allowed_task_time=allowed_task_time,
                        available_time=available_time,
                        current_intentions=current_intentions,
                        default_deadline=default_deadline,
                        default_time_est=default_time_est,
                        planning_fallacy_const=smdp_params["planning_fallacy_const"],
                        user_datetime=user_datetime,
                        min_sum_of_goal_values=min_sum_of_goal_values,
                        max_sum_of_goal_values=max_sum_of_goal_values,
                        min_goal_value_per_duration=min_goal_value_per_duration,
                        max_goal_value_per_duration=max_goal_value_per_duration,
                    )

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

                today_minutes = available_time[0]
                typical_minutes = available_time[1:]
                
                # Store anonymized to-do list in database
                log_dict["tree"] = delete_sensitive_data(projects)
                
                # Store today & typical daily minutes in database
                log_dict["today_minutes"] = today_minutes
                log_dict["typical_daily_minutes"] = typical_minutes

                # Store time: parsing to-do list
                timer["Generating to-do list"] = time.time() - tic

                # Start timer: parsing hierarchical structure
                tic = time.time()

                # New calculation + Save updated, user id, and skeleton
                try:
                    # Convert internal tree structure to items
                    flatten_projects = \
                        flatten_intentions(deepcopy(projects))
                    
                    # Anonymize and save flattened tree
                    log_dict["flatten_tree"] = \
                        delete_sensitive_data(flatten_projects)
                    
                    # Discard internal tree structure
                    leaf_projects = get_leaf_intentions(deepcopy(projects))
                    
                    # Anonymize and save tree with leaves
                    log_dict["leaf_tree"] = \
                        delete_sensitive_data(projects)
                    
                except:
                    status = "Something is wrong with your inputted " \
                             "goals and tasks. Please take a look at your " \
                             "Workflowy inputs and then try again."
                    
                    # Save the data if there was a change, removing nm fields so
                    # that we keep participant data anonymous
                    store_log(db.request_log, log_dict, status=status)
                    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)
                
                # Store time: parsing hierarchical structure
                timer["Parsing hierarchical structure"] = time.time() - tic
                
                # Start timer: storing parsed to-do list in database
                tic = time.time()

                # Save the data if there was a change, removing nm fields so
                # that we keep participant data anonymous
                store_log(db.request_log, log_dict, status="Save parsed tree")
                
                # Stop timer: storing parsed to-do list in database
                timer["Storing parsed to-do list in database"] = time.time() - tic
                
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
                        projects = assign_constant_points(leaf_projects,
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
                        projects = assign_length_points(leaf_projects)
                        
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
                            leaf_projects, distribution_fxn=distribution,
                            fxn_args=distribution_params)
                        
                    except:
                        status = "Problem while assigning points. "
    
                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)

                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)
                
                elif method == "smdp":
    
                    tic = time.time()
                    
                    final_tasks = assign_smdp_points(
                        projects, current_day=user_datetime, timer=timer,
                        day_duration=today_minutes,
                        all_json_items=flatten_projects,
                        smdp_params=smdp_params
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

                    timer["Run SMDP"] = time.time() - tic

                else:
                    status = "API method does not exist. Please contact us " \
                             "at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                # Start timer: Anonymizing data
                tic = time.time()
                
                # Update values in the tree
                log_dict["tree"] = delete_sensitive_data(projects)

                # Store time: Anonymizing date
                timer["Anonymize data"] = time.time() - tic

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
                
                # Store time: Storing incentivized tree in database
                timer["Storing incentivized tree in database"] = time.time() - tic

                if api_method == "updateTree":
                    cherrypy.response.status = 204
                    store_log(db.request_log, log_dict, status="Update tree")
                    return None
                
                elif api_method in \
                        {"getTasksForToday", "speedTest", "updateTransform"}:
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
                        
                        # Store time: Storing human-readable output
                        timer["Storing human-readable output"] = time.time() - tic
                        
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

                    # Store time: Storing successful pull in database
                    timer["Storing successful pull in database"] = time.time() - tic
                    
                    # print("\n===== Optimal items =====")
                    # for task in final_tasks:
                    #     # print(f"{task['nm']} & {task['val']} \\\\")
                    #     print(f"{task['nm']:100s} | {task['val']}")
                    # print()
                    
                    if api_method == "speedTest":
                        
                        status = f"The procedure took " \
                                 f"{time.time() - main_tic:.3f} seconds!"
    
                        # Stop timer: Complete SMDP procedure
                        timer["Complete SMDP procedure"] = \
                            time.time() - main_tic
    
                        return json.dumps({
                            "status": status,
                            "timer":  timer
                        })

                    else:
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
                    store_log(
                        db.request_log,
                        {"start_time": log_dict["start_time"],
                         "status":
                             status + " " +
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
