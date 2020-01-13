import cherrypy
import json
import os

from copy import deepcopy
from datetime import datetime
from pprint import pprint
from pymongo import MongoClient, DESCENDING

from src.apis import *
from src.schedulers import *
from src.point_scalers import utility_scaling
from src.utils import *

from todolistMDP.mdp_solvers import backward_induction, policy_iteration, \
                                    value_iteration
from todolistMDP.scheduling_solvers import simple_goal_scheduler

CONTACT = "Please contact the experimenter."


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
            return json.dumps({"status": status})
        
        # Can we load the request body (json)
        try:
            rawData = cherrypy.request.body.read()
            jsonData = json.loads(rawData)
        except:
            cherrypy.response.status = 403
            status = "No request body"
            return json.dumps({"status": status})
        return method(jsonData, *vpath, **params)


class PostResource(RESTResource):
    
    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):

        # Initialize log dictionary
        log_dict = {
            "start_time": datetime.now(),
        }
        
        try:
            # Compulsory parameters
            method = vpath[0]
            scheduler = vpath[1]
            user_key = vpath[-2]
            api_method = vpath[-1]
            
            # Additional parameters (the order of URL input matters!)
            parameters = [item for item in vpath[3:-3]]

            log_dict.update({
                "api_method": api_method,
                "duration": str(datetime.now() - log_dict["start_time"]),
                "method": method,
                "parameters": parameters,
                "scheduler": scheduler,
                "user_key": user_key,
                
                # Must be provided on each store (if needed)
                "allowed_task_time": None,
                "lm": None,
                "mixing_parameter": None,
                "status": None,
                "timestamp": None,
                "user_id": None,
            })
            
            # Get allowed task time | Default URL value: 'inf'
            try:
                allowed_task_time = float(vpath[2])
                log_dict["allowed_task_time"] = allowed_task_time
            except:
                status = "There was an issue with the allowed task time value."
                store_log(db.request_log, log_dict, status=status)
                cherrypy.response.status = 403
                return json.dumps({"status": status + " " + CONTACT})

            # Is there a user key
            try:
                current_id = jsonData["userkey"]
                log_dict["user_id"] = current_id
            except:
                status = "Problem with user key."
                store_log(db.request_log, log_dict, status=status)
                
                cherrypy.response.status = 403
                return json.dumps({"status": status + " " + CONTACT})

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
            if previous_result is not None:
                if jsonData["updated"] <= previous_result["lm"]:
                    status = "No update needed."
                    store_log(db.request_log, log_dict, status=status)

                    cherrypy.response.status = 403
                    return json.dumps({"status": status +
                        " If you think you are seeing this message in error, " +
                                                 CONTACT.lower()})

            # Update last modified
            log_dict["lm"] = jsonData["updated"]
            
            # Parse current intentions
            current_intentions = parse_current_intentions_list(
                jsonData["currentIntentionsList"])
            
            # New calculation + Save updated, user id, and skeleton
            try:
                projects = flatten_intentions(jsonData["projects"])
                log_dict["tree"] = create_projects_to_save(projects)
            except:
                status = "Error with parsing inputted projects."
                
                # Save the data if there was a change, removing nm fields so
                # that we keep participant data anonymous
                store_log(db.request_log, log_dict, status=status)
                
                cherrypy.response.status = 403
                return json.dumps({"status": status + " " + CONTACT})

            # Parse today hours
            try:
                today_hours = parse_hours(jsonData["today_hours"][0]["nm"])
                log_dict["today_hours"] = today_hours
            except:
                status = "Error with parsing today hours."
                store_log(db.request_log, log_dict, status=status)
                
                cherrypy.response.status = 403
                return json.dumps({"status": status + " " + CONTACT})
            
            try:
                typical_hours = parse_hours(jsonData["typical_hours"][0]["nm"])
                log_dict["typical_hours"] = typical_hours
            except:
                status = "Error with parsing typical hours."
                store_log(db.request_log, log_dict, status=status)
                
                cherrypy.response.status = 403
                return json.dumps({"status": status + " " + CONTACT})

            if not (0 < typical_hours <= 24):
                store_log(db.request_log, log_dict,
                          status="Invalid typical hours value.")
                
                status = "Please edit the hours you typically work today on Workflowy. " \
                         "The hours you work should be between 0 and 24."
                cherrypy.response.status = 403
                return json.dumps({"status": status})
            
            # 0 is an allowed value in case users want to skip a day
            if not (0 <= today_hours <= 24):
                store_log(db.request_log, log_dict,
                          status="Invalid today hours value.")
                
                status = "Please edit the hours you can work today on Workflowy. " \
                         "The hours you work should be between 0 and 24."
                cherrypy.response.status = 403
                return json.dumps({"status": status})
            
            # Convert typical and today hours into minutes
            typical_minutes = typical_hours * 60
            today_minutes = today_hours * 60
            
            # Subtract time estimation of current intentions from available time
            for task_id in current_intentions.keys():
                today_minutes -= current_intentions[task_id]["est"]

            # TODO: If it is necessary, check whether today_minutes > 0
            
            try:
                real_goals, misc_goals = \
                    parse_tree(projects, current_intentions, allowed_task_time,
                               today_minutes, typical_minutes)
            except Exception as error:
                status = str(error)
                
                # Remove personal data
                anonymous_error = parse_error_info(status)
                
                # Store error in DB
                store_log(db.request_log, log_dict, status=anonymous_error)
                
                status += " Please contact the experimenter if you feel you are not able to fix this issue."
                cherrypy.response.status = 403
                return json.dumps({"status": status})
            
            projects = real_goals + misc_goals
            log_dict["tree"] = create_projects_to_save(projects)

            # Save the data if there was a change, removing nm fields so that we
            # keep participant data anonymous
            store_log(db.request_log, log_dict, status="Save parsed tree")

            if previous_result is None:
                run_point_method = True
            else:
                run_point_method = are_there_tree_differences(
                    previous_result["tree"], projects)
            
            # Assign values for each task
            if run_point_method:
                if method == "constant":
                    projects = assign_constant_points(projects, *params)
                elif method == "random":
                    projects = assign_random_points(projects,
                                                    fxn_args=params)
                elif method == "length":
                    projects = assign_length_points(projects)
                
                # DP method
                elif method == "dp":
                    # Get mixing parameter | Default URL value: 0
                    mixing_parameter = int(vpath[3])
                    
                    # Convert the mixing parameter to probability
                    while mixing_parameter > 1:
                        mixing_parameter /= 10
                    
                    # Store the value of the mixing parameter in the log dict
                    log_dict['mixing_parameter'] = mixing_parameter

                    # Defined by the experimenter
                    if not (0 <= mixing_parameter < 1):
                        status = "There was an issue with the mixing-parameter value."
                        store_log(db.request_log, log_dict, status=status)
                        cherrypy.response.status = 403
                        return json.dumps({"status": status + " " + CONTACT})
                    
                    # TODO: Get informative exceptions
                    try:
                        final_tasks = \
                            assign_dynamic_programming_points(
                                real_goals, misc_goals,
                                solver_fn=simple_goal_scheduler,
                                scaling_fn=utility_scaling,
                                day_duration=today_minutes,
                                mixing_parameter=mixing_parameter,
                                verbose=False
                            )
                    except Exception as error:
                        cherrypy.response.status = 403
                        return json.dumps({"status": str(error)})
                
                # TODO: Test and fix potential bugs!
                elif method == "old-report":
                    final_tasks = \
                        assign_old_api_points(projects, backward_induction,
                                              duration=today_minutes)
                else:
                    status = "API method does not exist."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps({"status": status + " " + CONTACT})
            
            # If there are no changes in the tree, give back the values
            # assigned in the previous parsing
            else:
                for goal in projects:
                    corresponding_goal = (
                        next(item
                             for item in previous_result["tree"]
                             if item["id"] == goal["id"]))
                    for task in goal["ch"]:
                        task["val"] = (
                            next(item["val"]
                                 for item in corresponding_goal["ch"]
                                 if item["id"] == task["id"]))

            # Update values in the tree
            log_dict["tree"] = create_projects_to_save(projects)
            
            # Get task list from the tree
            task_list = task_list_from_projects(projects)
            
            # Schedule tasks for today
            if scheduler == "basic":
                final_tasks = basic_scheduler(
                    task_list, today_duration=today_minutes)
            elif scheduler == "deadline":
                final_tasks = deadline_scheduler(
                    task_list, today_duration=today_minutes)
            elif scheduler == "mdp":
                pass
            else:
                status = "Scheduling method does not exist."
                store_log(db.request_log, log_dict, status=status)
                cherrypy.response.status = 403
                return json.dumps({"status": status + " " + CONTACT})
            
            store_log(db.trees, log_dict, status="Save tree!")

            if api_method == "updateTree":
                cherrypy.response.status = 204
                store_log(db.request_log, log_dict, status="Update tree")
                return None
            
            elif api_method == "getTasksForToday":
                try:
                    final_tasks = clean_output(final_tasks)
                except:
                    status = "Error while preparing final output."
                    store_log(db.request_log, log_dict, status=status)
                    cherrypy.response.status = 403
                    return json.dumps({"status": status + " " + CONTACT})

                store_log(db.request_log, log_dict, status="Successful pull!")

                # Return scheduled tasks
                return json.dumps(final_tasks)
            
            else:
                status = "API Method not implemented."
                store_log(db.request_log, log_dict, status=status)
                
                cherrypy.response.status = 405
                return json.dumps({"status": status + " " + CONTACT})
            
        except Exception as error:
            status = "The API has encountered an error."
            
            # Store anonymous error info in DB collection
            anonymous_error = parse_error_info(str(error))
            store_log(db.request_log, log_dict,
                      status=status + " " + anonymous_error)
            
            cherrypy.response.status = 403
            return json.dumps({"status": status + " " + CONTACT})


class ExperimentPostResource(RESTResource):
    
    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        query = {'data.0.subject' : jsonData[0]["subject"]}
        if db.tiny_experiment.find(query).count() > 0:
            newvalues = { "$set": {"data":      jsonData,
                                       "timestamp": datetime.now()} }
            db.tiny_experiment.update_one(query, newvalues)
        else:
            db.tiny_experiment.insert_one({"data":      jsonData,
                                       "timestamp": datetime.now()})
        cherrypy.response.status = 204
        return None


class SurveyPostResource(RESTResource):

    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        query = {'data.0.url_variables.userid' : jsonData[0]["url_variables"]["userid"]}
        if db.tiny_survey.find(query).count() > 0:
            newvalues = { "$set": {"data":      jsonData,
                                       "timestamp": datetime.now()} }
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
