import os
import json_generator.nodes as nodes

from datetime import datetime
from json_generator.nodes import *
from json_generator.utils import *

# Branching factor
BRANCHING_FACTORS = [
    # 1,
    # 2,
    # 3,
    4,
    # 5,
]

# Tree depths
DEPTHS = [
    # 1,
    # 2,
    # 3,
    4,
    5,
    6,
    7
]

# Number of goals
N_GOALS = [
    1,
    2,
    3,
    4,
    5,
    # 6,
    # 7,
    # 8,
    # 9,
    # 10
]

# Today and typical working hours
HOURS_TODAY = 12
HOURS_TYPICAL = 12

# Type of input tree
INPUT_TYPE = "smdp"

# Path-related variables
PATH_NAME = f"data/{INPUT_TYPE}/"


def generate_file_name(n_goals, branching_factor, depth):
    return f"{n_goals}g_{branching_factor}bf_{depth}d.json"


def generate_multilevel_tree(node, branching_factor, depth, max_depth):
    
    # Stopping criterion
    if depth == max_depth:
        return
    
    # Generate next level
    new_nodes = node.generate_nodes(num_nodes=branching_factor, time_est_val=1)

    # Map generated goals
    map_nodes(node_id_dict, new_nodes)
    
    # Make recursive calls
    for ch_node in new_nodes:
        generate_multilevel_tree(
            node=ch_node, branching_factor=branching_factor,
            depth=depth + 1, max_depth=max_depth
        )


if __name__ == '__main__':
    
    for n_goals in N_GOALS:
        for branching_factor in BRANCHING_FACTORS:
            for max_depth in DEPTHS:
                
                n_tasks = branching_factor ** max_depth
    
                # Create path
                os.makedirs(PATH_NAME, exist_ok=True)
                
                # Generate file name
                FILE_NAME = PATH_NAME + generate_file_name(
                    n_goals, branching_factor, max_depth
                )
                
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
                deadline_val = "2099-01-01"
                
                # Generate goals
                new_nodes = node_id_dict[0].generate_nodes(
                    deadline_val=deadline_val, num_nodes=n_goals,
                    point_val=n_tasks, time_est_val=0
                )
                
                # Map generated goals
                map_nodes(node_id_dict, new_nodes)
                
                for node in new_nodes:
                    generate_multilevel_tree(node, branching_factor,
                                             depth=0, max_depth=max_depth)
                
                # Generate tasks for each goal
                # for node_id in [ch.get_id() for ch in ROOT_NODE.get_ch()]:
                #     new_nodes = \
                #         node_id_dict[node_id].generate_nodes(num_nodes=n_tasks,
                #                                              time_est_val=1)
                #
                # # Map generated tasks
                # map_nodes(node_id_dict, new_nodes)
                
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
                    }
                    
                    file.writelines(json.dumps(json_output, indent=4))
                    file.flush()
                
                print("Done!")
