import re
import cherrypy
import json
import os

goalCodeRegex = r"#CG(\d|&|_)"
totalValueRegex = r"(?:^| |>)\(?==(\d+)\)?(?:\b|$)"
timeEstimateRegex = r"(?:^| |>)\(?~~(\d+|\.\d+|\d+.\d+)(?:(h(?:ou)?(?:r)?)?(m(?:in)?)?)?s?\)?([^\da-z.]|$)"
deadlineRegex = r"DUE:(\d\d\d\d-\d\d-\d\d)(?:\b|$)"

def flatten_intentions(projects):
    for goal in projects:
        for child in goal["ch"]:
            if "ch" in child:
                goal["ch"].extend(child["ch"])
                del child["ch"]
    return projects

def parse_tree(projects):
    tree_structure = {}
    for goal in projects:
        #extract goal information
        goalCode = re.search(goalCodeRegex, goal["nm"], re.IGNORECASE)[1]
        value =  int(re.search(totalValueRegex, goal["nm"], re.IGNORECASE)[1])
        
        # #deal with misc goal
        # if goalCode == "_":
        #     goalCode = 0
        # else:
        #     goalCode = int(goalCode)
            
        #add goal to tree
        if goalCode not in tree_structure:
            tree_structure[goal["id"]] = []
            
        for child_idx, child in enumerate(goal["ch"]):
            time_est = re.search(timeEstimateRegex, child["nm"], re.IGNORECASE)
            if time_est[2] is not None:
                duration = 60*int(time_est[1])
            else:
                duration = int(time_est[1])
            
            tree_structure[goal["id"]].append([goalCode, goal["id"], child["id"], value, duration])
    return tree_structure


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
    print(method)
    if not method:
        methods = [x.replace("handle_", "") for x in dir(self) if x.startswith("handle_")]
        cherrypy.response.headers["Allow"] = ",".join(methods)
        raise cherrypy.HTTPError(405, "Method not implemented.")

    #can we load the request body (json)
    try:
        rawData = cherrypy.request.body.read()
        jsonData = json.loads(rawData)
    except:
        raise cherrypy.HTTPError(403, "No request body")
    return method(jsonData, *vpath, **params);

class PostResource(RESTResource):


    @cherrypy.tools.json_out()
    def handle_POST(self, jsonData, *vpath, **params):
        user_key, api_method = vpath
    
        #is there a user key
        try:
            current_id = jsonData["userkey"]
        except:
            raise cherrypy.HTTPError(403, "Problem with user key")
        if current_id != user_key:
            raise cherrypy.HTTPError(403, "Problem with user key")

        #check for changes if an existing user
        if current_id in cherrypy.session:
            if jsonData["updated"] <=  cherrypy.session[current_id]["updated"]:
                raise cherrypy.HTTPError(403, "No update needed")
        
        #new calculation
        #save updated, user id, and skeleton
        projects = flatten_intentions(jsonData["projects"])
        tree_structure = parse_tree(projects)

        if current_id not in cherrypy.session:
            cherrypy.session[current_id] = {}

        cherrypy.session[current_id]["updated"] = jsonData["updated"]
        cherrypy.session[current_id]["tree"] = tree_structure
        algorithm_output = json.dumps(tree_structure) 

        print(jsonData)
        print(tree_structure)
        if api_method == "updateTree":
            cherrypy.response.status = 204
            return None
        elif api_method == "getTasksForToday":
            #return scheduled tasks TODO finish this
            return algorithm_output
        else:
            raise cherrypy.HTTPError(405, "Method not implemented.")


class Root(object):
    tree = PostResource()

    @cherrypy.expose
    def index(self):
        return "REST example."


if __name__ == '__main__':
    conf = {
        '/tree': {
            'tools.sessions.on': True,
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        }
    }
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': int(os.environ.get('PORT', '5000'))})
    cherrypy.quickstart(Root(), '/', conf)