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
            methods = [x.replace("handle_", "") for x in dir(self) if
                       x.startswith("handle_")]
            cherrypy.response.headers["Allow"] = ",".join(methods)
            raise cherrypy.HTTPError(405, "Method not implemented.")
        
        # Can we load the request body (json)
        try:
            rawData = cherrypy.request.body.read()
            jsonData = json.loads(rawData)
        except:
            raise cherrypy.HTTPError(403, "No request body")
        
        return method(jsonData, *vpath, **params);


class PostResource(RESTResource):
    
    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        start_time = datetime.now()
        
        method = vpath[0]
        scheduler = vpath[1]
        parameters = [int(item) for item in vpath[2:-3]]
        user_key = vpath[-2]
        api_method = vpath[-1]
        
        # is there a user key
        try:
            current_id = jsonData["userkey"]
        except:
            raise cherrypy.HTTPError(403, "Problem with user key")
        # if current_id != user_key:
        #     raise cherrypy.HTTPError(403, "Problem with user key")
        
        if db.trees.find({'user_id': str(current_id)}) \
                   .sort('timestamp', DESCENDING).count() == 0:
            previous_result = 0
        else:
            previous_result = \
            db.trees.find({'user_id': str(current_id)}) \
                    .sort('timestamp', DESCENDING)[0]
        
        # check for changes if an existing user
        if previous_result != 0:
            if jsonData["updated"] <= previous_result["lm"]:
                raise cherrypy.HTTPError(403, "No update needed")
        
        # New calculation
        # Save updated, user id, and skeleton
        projects = flatten_intentions(jsonData["projects"])
        typical_hours = parse_hours(jsonData["typical_hours"][0]["nm"])
        today_hours = parse_hours(jsonData["today_hours"][0]["nm"])
        mixing_parameter = jsonData["mixing_parameter"]
        allowed_task_time = jsonData["allowed_task_time"]

        # TODO: What is the maximum amount of hours per day to work?
        if not (0 < typical_hours <= 24):
            raise cherrypy.HTTPError(403,
                        "The typical hours value has to be greater than 0!")
    
        # TODO: Maybe 0 <= today_hours (if users want to skip a day)?!
        # TODO: What is the maximum amount of hours per day to work?
        if not (0 < today_hours <= 24):
            raise cherrypy.HTTPError(403,
                        "The today hours value has to be greater than 0!")
        
        if not (0 <= mixing_parameter < 1):
            raise cherrypy.HTTPError(403,
                    "The mixing-parameter value has be in the interval [0, 1)")
        
        try:
            projects, missing_deadlines, missing_durations = \
                parse_tree(projects, allowed_task_time)
        except cherrypy.HTTPError as error:
            raise cherrypy.HTTPError(403, error)
        
        if previous_result == 0:
            run_point_method = True
        else:
            run_point_method = are_there_tree_differences(
                previous_result["tree"], projects)
        
        if run_point_method or (scheduler == "mdp"):  # TODO if we can do scheduling with old MDP points, we should do that
            if method == "constant":
                projects = assign_constant_points(projects, *parameters)
            elif method == "random":
                projects = assign_random_points(projects, fxn_args=parameters)
            elif method == "hierarchical":
                projects = assign_hierarchical_points(projects)
            elif method == "length":
                projects = assign_length_points(projects)
            elif method == "old-report":
                final_tasks = \
                    assign_old_api_points(projects, simple_goal_scheduler,
                                          duration=today_hours * 60,
                                          mixing_parameter=mixing_parameter)
            else:
                raise cherrypy.HTTPError(403, "API method does not exist")
        else:
            # join old vals to projects
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
            raise cherrypy.HTTPError(403, "Scheduling method does not exist")
        
        # save the data if there was a change, removing nm fields so that we
        # keep participant data anonymous
        if run_point_method:
            save_projects = deepcopy(projects)
            for project in save_projects:
                del project["nm"]
                for task in project["ch"]:
                    del task["nm"]
            
            db.trees.insert_one(
                {"user_id": current_id, "timestamp": datetime.now(),
                 "duration": str(datetime.now() - start_time),
                 "lm": jsonData["updated"], "tree": save_projects})
        
        if api_method == "updateTree":
            cherrypy.response.status = 204
            return None
        elif api_method == "getTasksForToday":
            # return scheduled tasks
            final_tasks = clean_output(final_tasks)
            return json.dumps(final_tasks)
        else:
            raise cherrypy.HTTPError(405, "Method not implemented.")


class Root(object):
    api = PostResource()
    
    @cherrypy.expose
    def index(self):
        return "REST API for Complice Project w/ Workflowy Points"


if __name__ == '__main__':
    os.environ[
        'MONGODB_URI'] = "mongodb://heroku_g6l4lr9d:g9q8u1oon9naso4ncfcokgtlkm@ds341557.mlab.com:41557/heroku_g6l4lr9d"
    conn = MongoClient(os.environ['MONGODB_URI'] + "?retryWrites=false")
    db = conn.heroku_g6l4lr9d
    
    conf = {
        '/': {
            # 'tools.sessions.on': True,
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        }
    }
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update(
        {'server.socket_port': int(os.environ.get('PORT', '6789'))})
    cherrypy.quickstart(Root(), '/', conf)
