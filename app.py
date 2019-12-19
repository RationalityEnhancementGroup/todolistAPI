import cherrypy
import json
import os

from src.utils import *
from src.apis import *
from src.schedulers import *
from pymongo import MongoClient, DESCENDING
from datetime import datetime
from copy import deepcopy


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
            return json.dumps({"status":"Method not implemented."})
    
        # Can we load the request body (json)
        try:
            rawData = cherrypy.request.body.read()
            jsonData = json.loads(rawData)
        except:
            cherrypy.response.status = 403
            return json.dumps({"status":"No request body"})
        return method(jsonData, *vpath, **params)


class PostResource(RESTResource):
    
    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        try:
            start_time = datetime.now()
            
            method = vpath[0]
            scheduler = vpath[1]
            allowed_task_time = np.float("inf")# TODO: Check URL allowed_task_time parameter value
            parameters = [int(item) for item in vpath[2:-3]]
            user_key = vpath[-2]
            api_method = vpath[-1]

            
            # is there a user key
            try:
                current_id = jsonData["userkey"]
                db.request_log.insert_one(
                    {"user_id": current_id, "method" : method,"scheduler" : scheduler,"parameters" :parameters,"user_key" :user_key,"api_method" :api_method,"timestamp": datetime.now()})
            
            except:
                db.request_log.insert_one(
                    {"user_id": "null", "method" : method,"scheduler" : scheduler,"parameters" :parameters,"user_key" :user_key,"api_method" :api_method,"timestamp": datetime.now()})
            
                cherrypy.response.status = 403
                return json.dumps({"status": "Problem with user key. Please contact the experimenter."})

            
            if db.trees.find({'user_id': str(current_id)}) \
                       .sort('timestamp', DESCENDING).count() == 0:
                previous_result = 0
            else:
                previous_result = \
                db.trees.find({'user_id': str(current_id)}) \
                        .sort('timestamp', DESCENDING)[0]
            
            # Check for changes if an existing user (..?)
            if previous_result != 0:
                if jsonData["updated"] <=  previous_result["lm"]:
                    cherrypy.response.status = 403
                    return json.dumps({"status":"No update needed. If you think you are seeing this message in error, please contact the experimenter."})
            
            #new calculation
            #save updated, user id, and skeleton
            try:
                projects = flatten_intentions(jsonData["projects"])
            except:
                cherrypy.response.status = 403
                return json.dumps({"status":"Error with parsing inputted projects. Please contact the experimenter."})
            try:
                typical_hours = parse_hours(jsonData["typical_hours"][0]["nm"])
                today_hours = parse_hours(jsonData["today_hours"][0]["nm"])
            except:
                cherrypy.response.status = 403
                return json.dumps({"status": "Error with parsing inputted hours. Please contact the experimenter."})

            if not (0 < typical_hours <= 24):
                cherrypy.response.status = 403
                return json.dumps({"status": "Please edit the hours you typically work today on Workflowy. The hours you work should be between 0 and 24."})
        
            # 0 is an allowed value in case users want to skip a day
            if not (0 <= today_hours <= 24):
                cherrypy.response.status = 403
                return json.dumps({"status": "Please edit the hours you can work today on Workflowy. The hours you work should be between 0 and 24."})


            # New calculation
            # Save updated, user id, and skeleton
            try:
                projects = flatten_intentions(jsonData["projects"])
            except:
                cherrypy.response.status = 403
                return json.dumps({"status":
                                       "Error with parsing inputted projects. Please contact the experimenter."})

            try:
                real_goals, misc_goals = parse_tree(projects, allowed_task_time,
                                                    typical_hours)
            except Exception as error:  # TODO: Write specific exceptions
                cherrypy.response.status = 403
                return json.dumps({"status": str(error) + " Please contact the experimenter if you feel you are not able to fix this issue."})
            
            projects = real_goals + misc_goals

            if previous_result == 0:
                run_point_method = True
            else:
                run_point_method = are_there_tree_differences(
                    previous_result["tree"], projects)

            # TODO if we can do scheduling with old MDP points, we should do that
            if run_point_method or (scheduler == "mdp") or (scheduler == "dp"):
                if method == "constant":
                    projects = assign_constant_points(projects, *parameters)
                elif method == "random":
                    projects = assign_random_points(projects, fxn_args=parameters)
                elif method == "hierarchical":
                    projects = assign_hierarchical_points(projects)
                elif method == "length":
                    projects = assign_length_points(projects)
                    
                # DP method
                elif method == "dp":
                    # TODO: URL input 
                    mixing_parameter = parameters[-1]

                    # TODO: Edit after making it URL input
                    # Defined by the experimenter
                    if not (0 <= mixing_parameter < 1):
                        cherrypy.response.status = 403
                        return json.dumps({"status": "Please contact the experimenter. There was an issue with the mixing-parameter value."})

                    try:
                        final_tasks = \
                            assign_dynamic_programming_points(
                                real_goals, misc_goals, simple_goal_scheduler,
                                day_duration=today_hours * 60,
                                mixing_parameter=mixing_parameter,
                                verbose=False
                            )
                    except Exception as error:
                        cherrypy.response.status = 403
                        return json.dumps({"status": str(error)})
                    
                elif method == "old-report":
                    final_tasks = \
                        assign_old_api_points(projects, backward_induction,
                                              duration=today_hours * 60)
                else:
                    cherrypy.response.status = 403
                    return json.dumps({"status": "API method does not exist. Please contact the experimenter."})
            else:
                # Join old values to projects
                for project in projects:
                    corresponding_goal = (next(
                        item for item in previous_result["tree"] if
                        item["id"] == project["id"]))
                    for task in project["ch"]:
                        task["val"] = (next(
                            item["val"] for item in corresponding_goal["ch"] if
                            item["id"] == task["id"]))
            
            task_list = task_list_from_projects(projects)
            if scheduler == "basic":
                final_tasks = basic_scheduler(task_list,
                                              today_duration=today_hours * 60)
            elif scheduler == "deadline":
                final_tasks = deadline_scheduler(task_list,
                                                 today_duration=today_hours * 60)
            elif scheduler == "mdp":
                pass
            else:
                cherrypy.response.status = 403
                return json.dumps({"status": "Scheduling method does not exist. Please contact the experimenter."})

            # TODO: Make this function @ utils.py
            # Save the data if there was a change, removing nm fields so that we
            # keep participant data anonymous
            if run_point_method:
                save_projects = deepcopy(projects)
                for project in save_projects:
                    del project["nm"]
                    for task in project["ch"]:
                        del task["nm"]
                
                db.trees.insert_one(
                    {"user_id": current_id, "method" : method,"scheduler" : scheduler,"parameters" :parameters,"user_key" :user_key,"api_method" :api_method,"timestamp": datetime.now(),
                     "duration": str(datetime.now() - start_time),
                     "lm": jsonData["updated"], "tree": save_projects})
            
            if api_method == "updateTree":
                cherrypy.response.status = 204
                return None
            elif api_method == "getTasksForToday":
                # Return scheduled tasks
                final_tasks = clean_output(final_tasks)
                return json.dumps(final_tasks)
            else:
                cherrypy.response.status = 405
                return json.dumps({"status": "API Method not implemented. Please contact the experimenter."})
        except:
            cherrypy.response.status = 403
            return json.dumps({"status": "Please contact the experimenter. The API has encountered an error."})


class ExperimentPostResource(RESTResource):

    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        db.tiny_experiment.insert_one({"data": jsonData,
                                       "timestamp":  datetime.now()})
        cherrypy.response.status = 204
        return None

class SurveyPostResource(RESTResource):


    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        db.tiny_survey.insert_one({"data": jsonData, "timestamp":  datetime.now()})
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
        '/': {
            # 'tools.sessions.on': True,
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]},
        '/static':{
            'tools.staticdir.on': True,
            'tools.staticdir.dir': os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'static'),
            'tools.staticdir.index': 'instructions/experiment_instructions.html'
            }
        }

    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update(
        {'server.socket_port': int(os.environ.get('PORT', '6789'))})
    cherrypy.quickstart(Root(), '/', conf)
