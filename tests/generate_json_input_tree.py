import os
import json_generator.nodes as nodes

from datetime import datetime
from json_generator.nodes import *
from json_generator.utils import *

# Number of goals
# N_GOALS = [10, 50, 100, 500, 1000, 2500, 5000, 7500, 10000]
# N_GOALS = [1]
N_GOALS = list(range(2, 11))

# Number of tasks
# N_TASKS = [1]
# N_TASKS = [10, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]
# N_TASKS = [25, 50, 75, 100, 125, 150, 250, 500, 750, 1000]
N_TASKS = [1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000,
           3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000]

# Today and typical working hours
HOURS_TODAY = 12
HOURS_TYPICAL = 12

# Deadlines
# DEADLINES = [None]
TODAY_DATE = datetime.today().date()
# DEADLINE_YEARS = [1, 2, 3, 5, 10, 15, 20]
DEADLINE_YEARS = [1]

# Type of input tree
INPUT_TYPE = "smdp"

# Path-related variables
PATH_NAME = f"data/{INPUT_TYPE}/"


def generate_file_name(n_goals, n_tasks, years):
    return f"{n_goals}_goals_{n_tasks}_tasks_{years}_years.json"


for n_goals in N_GOALS:
    for n_tasks in N_TASKS:
        for years in DEADLINE_YEARS:
            
            # Create path
            os.makedirs(PATH_NAME, exist_ok=True)
            
            # Generate file name | TODO: Find a smarter way to generate names
            FILE_NAME = PATH_NAME + generate_file_name(n_goals, n_tasks, years)
            
            print(f"Generating {FILE_NAME}...", end=" ")
            
            # Reset node counter
            nodes.NODE_COUNTER = 0
            
            # Generate root node (the node that connects all the goal nodes
            ROOT_NODE = Node(id=0)
            node_id_dict = {
                0: ROOT_NODE
            }
            
            # Set root ID to be none
            ROOT_NODE.id = "None"
            
            # Generate deadline value
            deadline_val = \
                f"{TODAY_DATE.year + years}-{TODAY_DATE.month}-{TODAY_DATE.day}"
            
            # Generate goals
            new_nodes = node_id_dict[0].generate_nodes(
                deadline_val=deadline_val, num_nodes=n_goals,
                point_val=n_tasks, time_est_val=0
            )
            
            # Map generated goals
            map_nodes(node_id_dict, new_nodes)
            
            # Generate tasks for each goal
            for node_id in [ch.get_id() for ch in ROOT_NODE.get_ch()]:
                new_nodes = \
                    node_id_dict[node_id].generate_nodes(num_nodes=n_tasks,
                                                         time_est_val=1)
            
            # Map generated tasks
            map_nodes(node_id_dict, new_nodes)
            
            # Save JSON tree
            with open(FILE_NAME, "w") as file:
                json_output = {
                    "currentIntentionsList": [],
                    "projects":              ROOT_NODE.generate_tree(),
                    "timezoneOffsetMinutes": 0,
                    "today_hours":           [
                        {
                            "id": "_",
                            "nm": f"#HOURS_TODAY =={HOURS_TODAY}",
                            "lm": 0
                        }
                    ],
                    "typical_hours":         [
                        {
                            "id": "_",
                            "nm": f"#HOURS_TYPICAL =={HOURS_TYPICAL}",
                            "lm": 0
                        }
                    ],
                    "updated":               0,
                    "userkey":               "__test__",
                    "n_goals":               n_goals,
                    "n_tasks":               n_tasks
                }
                
                file.writelines(json.dumps(json_output, indent=4))
                file.flush()
            
            print("Done!")
