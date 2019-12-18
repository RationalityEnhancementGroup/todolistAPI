def add_new_goals_and_tasks(root_node, goals, tasks):
    goal_nodes = root_node.generate_nodes(goals)
    for i in range(len(goal_nodes)):
        goal_nodes[i].generate_nodes(tasks[i])
    return


def update_node_id_dict(dct, nodes):
    for node in nodes:
        if node.id in dct.keys():
            print(f'ID {node.id} exists!')
        else:
            dct[node.id] = node
    return dct


def map_nodes(node_id_dict, nodes):
    for node in nodes:
        if node.id in node_id_dict.keys():
            print(f'ID {node.id} already exists!')
        else:
            node_id_dict[node.id] = node
            for ch in node.ch:
                map_nodes(ch)
    return
