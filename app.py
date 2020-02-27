import cherrypy
import json
import os
import stopit

from copy import deepcopy
from datetime import datetime
from pprint import pprint
from pymongo import MongoClient, DESCENDING

from src.apis import *
from src.schedulers import *
from src.point_scalers import utility_scaling
from src.utils import *

from todolistMDP.mdp_solvers \
    import backward_induction, policy_iteration, value_iteration
from todolistMDP.scheduling_solvers \
    import run_dp_algorithm, run_greedy_algorithm

CONTACT = "If you continue to encounter this issue, please contact us at reg.experiments@tuebingen.mpg.de."
TIMEOUT_SECONDS = 28


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

        with stopit.ThreadingTimeout(TIMEOUT_SECONDS) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING

            # Initialize log dictionary
            log_dict = {
                "start_time": datetime.now(),
            }

            try:
                # Compulsory parameters
                method = vpath[0]
                scheduler = vpath[1]
                default_duration = vpath[2]  # This needs to be in minutes
                default_deadline = vpath[3]  # This needs to be in days
                allowed_task_time = vpath[4]
                exp_identifier = vpath[-4]
                user_key = vpath[-2]
                api_method = vpath[-1]

                if "cite" in exp_identifier:
                    round_param = 2
                else:
                    round_param = 0
                
                # Additional parameters (the order of URL input matters!)
                # Casting to other data types is done in the functions that use
                # these parameters
                parameters = [item for item in vpath[5:-4]]

                log_dict.update({
                    "api_method": api_method,
                    "allowed_task_time": allowed_task_time,
                    "duration": str(datetime.now() - log_dict["start_time"]),
                    "exp_identifier": exp_identifier,
                    "method": method,
                    "parameters": parameters,
                    "round_param": round_param,
                    "scheduler": scheduler,
                    "user_key": user_key,
                    
                    # Must be provided on each store (if needed)
                    "lm": None,
                    "mixing_parameter": None,
                    "status": None,
                    "timestamp": None,
                    "user_id": None,
                })
                
                # Get allowed task time | Default URL value: 'inf'
                try:
                    allowed_task_time = float(allowed_task_time)
                    log_dict["allowed_task_time"] = allowed_task_time
                except:
                    status = "There was an issue with the API input (allowed time parameter.) Please contact us at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)

                # Is there a user key
                try:
                    current_id = jsonData["userkey"]
                    log_dict["user_id"] = current_id
                except:
                    status = "We encountered a problem with the inputs from Workflowy, please try again."
                    store_log(db.request_log, log_dict, status=status)
                    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)

                # Get the latest result from the current user
                try:
                    previous_result = \
                        db.request_log.find({"user_id": str(current_id),
                                             "status":  "Successful pull!"}) \
                                      .sort("timestamp", DESCENDING)[0]
                # If this is a new user, then run the value assignment
                except:
                    previous_result = None
                    
                # Check for changes if an existing user (..?)
                if (previous_result is not None) and \
                        (len(jsonData["currentIntentionsList"]) > 0):
                    if jsonData["updated"] <= previous_result["lm"]:
                        status = "No update needed. If you want to pull a specific task, please tag it #today on Workflowy."
                        store_log(db.request_log, log_dict, status=status)

                        cherrypy.response.status = 403
                        return json.dumps(status + " " + CONTACT)

                # Update last modified
                log_dict["lm"] = jsonData["updated"]
                
                # Parse current intentions
                try:
                    current_intentions = parse_current_intentions_list(
                        jsonData["currentIntentionsList"], default_duration=default_duration)

                except:
                    status = "An error related to the current intentions has occurred."
                    
                    # Save the data if there was a change, removing nm fields so
                    # that we keep participant data anonymous
                    store_log(db.request_log, log_dict, status=status)

                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)

                # New calculation + Save updated, user id, and skeleton
                try:
                    projects = flatten_intentions(jsonData["projects"])
                    log_dict["tree"] = create_projects_to_save(projects)
                except:
                    status = "Something is wrong with your inputted goals and tasks. Please take a look at your Workflowy inputs and then try again."
                    
                    # Save the data if there was a change, removing nm fields so
                    # that we keep participant data anonymous
                    store_log(db.request_log, log_dict, status=status)
                    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)

                # Parse today hours
                try:
                    today_hours = parse_hours(jsonData["today_hours"][0]["nm"])
                    log_dict["today_hours"] = today_hours
                except:
                    status = "Something is wrong with the hours in HOURS_TODAY. Please take a look and try again."
                    store_log(db.request_log, log_dict, status=status)
                    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)
                
                try:
                    typical_hours = parse_hours(jsonData["typical_hours"][0]["nm"])
                    log_dict["typical_hours"] = typical_hours
                except:
                    status = "Something is wrong with the hours in HOURS_TYPICAL. Please take a look and try again."
                    store_log(db.request_log, log_dict, status=status)
                    
                    cherrypy.response.status = 403
                    return json.dumps(status + " " + CONTACT)

                if not (0 < typical_hours <= 24):
                    store_log(db.request_log, log_dict,
                              status="Invalid typical hours value.")
                    
                    status = "Please edit the hours you typically work on Workflowy. " \
                             "The hours you work should be between 0 and 24."
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                # 0 is an allowed value in case users want to skip a day
                if not (0 <= today_hours <= 24):
                    store_log(db.request_log, log_dict,
                              status="Invalid today hours value.")
                    
                    status = "Please edit the hours you can work today on Workflowy. " \
                             "The hours you work should be between 0 and 24."
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                # Convert typical and today hours into minutes
                typical_minutes = typical_hours * 60
                today_minutes = today_hours * 60
                
                # Subtract time estimation of current intentions from available time
                for task_id in current_intentions.keys():
                    today_minutes -= current_intentions[task_id]["est"]

                # TODO: If it is necessary, check whether today_minutes > 0
                
                try:
                    real_goals, misc_goals = \
                        parse_tree(projects, current_intentions,
                                   allowed_task_time,
                                   today_minutes, typical_minutes,
                                   default_duration=int(default_duration),
                                   default_deadline=int(default_deadline))
                except Exception as error:
                    status = str(error)
                    
                    # Remove personal data
                    anonymous_error = parse_error_info(status)
                    
                    # Store error in DB
                    store_log(db.request_log, log_dict, status=anonymous_error)
                    
                    status += " Please take a look at your Workflowy inputs and then try again. "
                    cherrypy.response.status = 403
                    return json.dumps(status + CONTACT)
                
                projects = real_goals + misc_goals
                log_dict["tree"] = create_projects_to_save(projects)

                # Save the data if there was a change, removing nm fields so
                # that we keep participant data anonymous
                store_log(db.request_log, log_dict, status="Save parsed tree")

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

                # DP method
                elif method == "dp" or method == "greedy":
                    # Get mixing parameter | Default URL value: 0
                    mixing_parameter = float('0.' + parameters[0])
                    
                    # Store the value of the mixing parameter in the log dict
                    log_dict['mixing_parameter'] = mixing_parameter

                    # Defined by the experimenter
                    if not (0 <= mixing_parameter < 1):
                        status = "There was an issue with the API input (mixing-parameter value). Please contact us at reg.experiments@tuebingen.mpg.de."
                        store_log(db.request_log, log_dict, status=status)
                        cherrypy.response.status = 403
                        return json.dumps(status)

                    utility_inputs = {'scale_min': None, 'scale_max': None}
                    if len(parameters) >= 3:
                        utility_inputs['scale_min'] = float(parameters[1])
                        utility_inputs['scale_max'] = float(parameters[2])

                    # Use function default, if not in URL
                    if len(parameters) >= 4:
                        utility_inputs['scaling_fn'] = parameters[3]
                        
                    solver_fn = run_dp_algorithm
                    if method == "greedy":
                        solver_fn = run_greedy_algorithm
                        
                    # TODO: Get informative exceptions
                    try:
                        final_tasks = \
                            assign_dynamic_programming_points(
                                real_goals, misc_goals,
                                solver_fn=solver_fn,
                                scaling_fn=utility_scaling,
                                scaling_inputs=utility_inputs,
                                day_duration=today_minutes,
                                mixing_parameter=mixing_parameter,
                                verbose=False
                            )
                    except Exception as error:
                        cherrypy.response.status = 403
                        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:

                            error = "The API took too long retrieving your Workflowy information, please try again."
            
                        return json.dumps(str(error) + " " + CONTACT)
                
                # TODO: Test and fix potential bugs!
                elif method == "old-report":
                    final_tasks = \
                        assign_old_api_points(projects, backward_induction,
                                              duration=today_minutes)
                else:
                    status = "API method does not exist. Please contact us at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
        
                # Update values in the tree
                log_dict["tree"] = create_projects_to_save(projects)
                
                # Get task list from the tree
                task_list = task_list_from_projects(projects)
                
                # Schedule tasks for today
                if scheduler == "basic":
                    try:
                        final_tasks = basic_scheduler(
                            task_list, today_duration=today_minutes)
                    except Exception as error:
                        status = str(error) + ' '
    
                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)
    
                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)

                elif scheduler == "deadline":
                    try:
                        final_tasks = deadline_scheduler(
                            task_list, today_duration=today_minutes)
                    except Exception as error:
                        status = str(error) + ' '

                        # Store error in DB
                        store_log(db.request_log, log_dict, status=status)

                        cherrypy.response.status = 403
                        return json.dumps(status + CONTACT)

                elif scheduler == "mdp":
                    pass
                else:
                    status = "Scheduling method does not exist. Please contact us at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps(status)
                
                store_log(db.trees, log_dict, status="Save tree!")

                if api_method == "updateTree":
                    cherrypy.response.status = 204
                    store_log(db.request_log, log_dict, status="Update tree")
                    return None
                
                elif api_method == "getTasksForToday":
                    if len(final_tasks) == 0:
                        status = "No update needed. If you want to pull a specific task, please tag it #today on Workflowy."
                        store_log(db.request_log, log_dict, status=status)
                        cherrypy.response.status = 403
                        return json.dumps(status + " " + CONTACT)
                    try:
                        final_tasks = clean_output(final_tasks, round_param)
                    except:
                        status = "Error while preparing final output."
                        store_log(db.request_log, log_dict, status=status)
                        cherrypy.response.status = 403
                        return json.dumps(status + " " + CONTACT)

                    store_log(db.request_log, log_dict, status="Successful pull!")

                    # Return scheduled tasks
                    return json.dumps(final_tasks)
                
                else:
                    status = "API Method not implemented. Please contact us at reg.experiments@tuebingen.mpg.de."
                    store_log(db.request_log, log_dict, status=status)
                    
                    cherrypy.response.status = 405
                    return json.dumps(status)
                
            except Exception as error:
                status = "The API has encountered an error, please try again."
                
                # Store anonymous error info in DB collection
                anonymous_error = parse_error_info(str(error))
                store_log(db.request_log, log_dict,
                          status=status + " " + anonymous_error)
                
                cherrypy.response.status = 403
                return json.dumps(status + " " + CONTACT)


class ExperimentPostResource(RESTResource):
    
    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        query = {'data.0.subject': jsonData[0]["subject"]}
        if db.tiny_experiment.find(query).count() > 0:
            newvalues = { "$set": {"data":      jsonData,
                                   "timestamp": datetime.now()}}
            db.tiny_experiment.update_one(query, newvalues)
        else:
            db.tiny_experiment.insert_one({"data":      jsonData,
                                           "timestamp": datetime.now()})
        cherrypy.response.status = 204
        return None


class SurveyPostResource(RESTResource):

    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        query = {'data.0.url_variables.userid':
                     jsonData[0]["url_variables"]["userid"]}
        if db.tiny_survey.find(query).count() > 0:
            newvalues = { "$set": {"data":      jsonData,
                                   "timestamp": datetime.now()}}
            db.tiny_survey.update_one(query, newvalues)
        else:
            db.tiny_survey.insert_one({"data":      jsonData,
                                       "timestamp": datetime.now()})
        cherrypy.response.status = 204
        return None


class Root(object):
    api = PostResource()
    experiment_data = ExperimentPostResource()
    survey_data = SurveyPostResource()
    
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
            'tools.staticdir.index': 'instructions/experiment_instructions.html'
        }
    }
    
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update(
        {'server.socket_port': int(os.environ.get('PORT', '6789'))})
    cherrypy.quickstart(Root(), '/', conf)
