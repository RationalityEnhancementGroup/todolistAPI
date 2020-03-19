from json_generator.nodes import *
from json_generator.value_generator import *
from json_generator.utils import *
from pprint import pprint


""" Generate root node of the tree """
# Generate root node (the node that connects all the goal (level 1) nodes
ROOT_NODE = Node(id=0)
node_id_dict = {
    0: ROOT_NODE
}

""" (+) Manual node generation
 
    Use the DEFAULT_* values in order to avoid getting None in the nodes' name!
"""
new_nodes = node_id_dict[0].generate_nodes(
    points=[DEFAULT_POINTS for _ in range(3)]
)
map_nodes(node_id_dict, new_nodes)

new_nodes = node_id_dict[3].generate_nodes(
    points=[DEFAULT_POINTS for _ in range(3)]
)
map_nodes(node_id_dict, new_nodes)

new_nodes = node_id_dict[6].generate_nodes(
    points=[DEFAULT_POINTS for _ in range(3)]
)
map_nodes(node_id_dict, new_nodes)

new_nodes = node_id_dict[9].generate_nodes(
    points=[DEFAULT_POINTS for _ in range(3)]
)
map_nodes(node_id_dict, new_nodes)

# pprint(node_id_dict)


""" (+) Point values
        (+) Generate related point values
        (+) Generate reversely-related point values
        (+) Generate random point values
"""
generate_points(node_id_dict[1], fn=generate_related_values,
                proposed_points=np.arange(100), reverse=False)

generate_points(node_id_dict[2], fn=generate_related_values,
                proposed_points=np.arange(100), reverse=True)

generate_points(node_id_dict[3], fn=np.random.randint, low=10, high=100)

""" (+) Time-estimation values
        (+) Generate uniform time-estimation value between goals
        (+) Generate uniform time-estimation value within a goal
        (+) Change time-estimation value only for one goal
        (+) Generate random time-estimation values
"""
generate_time_est(ROOT_NODE, value=1000)
for id in [3, 6]:
    generate_time_est(node_id_dict[id], value=100)
generate_time_est(node_id_dict[6], value=500)
generate_time_est(node_id_dict[1], np.random.randint, low=10, high=100)

""" (+) Set points manually """
node_id_dict[1].set_points(100)
node_id_dict[2].set_points(300)

""" (+) Set time-estimation manually """
node_id_dict[10].set_time_est(200)
node_id_dict[11].set_time_est(300)
node_id_dict[12].set_time_est(100)

""" (+) Generate JSON tree """
print(json.dumps(ROOT_NODE.generate_tree(), indent=4))
