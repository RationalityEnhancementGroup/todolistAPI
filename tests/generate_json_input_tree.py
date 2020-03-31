import os
import json_generator.nodes as nodes

from json_generator.nodes import *
from json_generator.utils import *

# Number of goals
N_GOALS = [2]
# N_GOALS = [10]

# Number of tasks
N_TASKS = [10]
# N_TASKS = [10, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]

# Today and typical working hours
HOURS_TODAY = 24
HOURS_TYPICAL = 24

# Type of input tree
INPUT_TYPE = "tags"

# Path-related variables
PATH_NAME = f"tests/data/{INPUT_TYPE}/"
FILE_NAME = None

for n_goals in N_GOALS:
    for n_tasks in N_TASKS:
        
        # Create path
        os.makedirs(PATH_NAME, exist_ok=True)
        
        # Generate file name
        if FILE_NAME is None:
            FILE_NAME = PATH_NAME + f"{n_goals}_goals_{n_tasks}_tasks.json"
        else:
            FILE_NAME = PATH_NAME + FILE_NAME
        
        print(f"Generating {FILE_NAME}...", end=" ")
    
        # Reset node counter
        nodes.NODE_COUNTER = 0
    
        # Generate root node (the node that connects all the goal nodes
        ROOT_NODE = Node(id=0)
        node_id_dict = {
            0: ROOT_NODE
        }
        
        # Generate goals
        new_nodes = node_id_dict[0].generate_nodes(num_nodes=n_goals,
                                                   point_val=n_tasks,
                                                   time_est_val=0)
    
        # Map generated goals
        map_nodes(node_id_dict, new_nodes)
        
        # Generate tasks for each goal
        for node_id in [ch.get_id() for ch in ROOT_NODE.get_ch()]:
            new_nodes = node_id_dict[node_id].generate_nodes(num_nodes=n_tasks,
                                                             time_est_val=1)
        
        # Map generated tasks
        map_nodes(node_id_dict, new_nodes)
        
        # Save JSON tree
        with open(FILE_NAME, "w") as file:
            json_output = {
                "currentIntentionsList": [],
                "projects": ROOT_NODE.generate_tree(),
                "timezoneOffsetMinutes": 0,
                "today_hours": [
                    {
                        "id": "_",
                        "nm": f"#HOURS_TODAY =={HOURS_TODAY}",
                        "lm": 0
                    }
                ],
                "typical_hours": [
                    {
                        "id": "_",
                        "nm": f"#HOURS_TYPICAL =={HOURS_TYPICAL}",
                        "lm": 0
                    }
                ],
                "updated": 0,
                "userkey": "__test__"
            }
            
            file.writelines(json.dumps(json_output, indent=4))
            file.flush()
        
        print("Done!")
