import numpy as np


def generate_points(node, fn, update=True, **dist_params):
    """
    Generates points in for its children nodes in one of the following manners:
    - From a set of values w(/o) repetition |
      fn: np.random.choice(a, size, replace)
    - Randomly | fn: np.random.<distribution>(<distribution_parameters>, size)
        - The distribution needs to be specified
    - Related to the sum of the time estimations of the children nodes |
      fn: generate_related_values(node, proposed_points, reverse)
        - Options
            - In accordance with the time estimations
            - Opposite to the time estimations
    
    Args:
        node: Parent node of the nodes for which points are generated.
        fn: Function that generates points.
        update: Whether to update children nodes or not.
        **dist_params: Parameters for the function that generates points.

    Returns:
        List of points.
    """
    size = len(node)  # Number of children nodes of the provided node
    
    if fn == generate_related_values:
        points = generate_related_values(node=node, **dist_params)
    else:
        # Any other function (e.g. np.random.*)
        points = fn(**dist_params, size=size)
    
    if update:
        for i in range(size):
            node.ch[i].set_points(points[i])

    return points


def generate_related_values(node, proposed_points, reverse=False):
    """
    Generates point values for the children (in this case, they have to be leaf)
    nodes of the provided node, based on the values of their time estimations.
    
    Args:
        node: Node whose point values of its children going to be changed.
        proposed_points: List of proposed points. The number of unique values in
                         this list should be larger than the unique
                         time-estimation values.
        reverse: Whether the generated values to be in accordance with the
                 time-estimation values or opposite to them.

    Returns:
        Array of point values related to the time-estimation values.
    """
    # Get unique time-estimation values
    time_est = [ch.time_est for ch in node.get_ch()]
    unique_time_ests = get_unique_values(time_est)
    
    # Get unique proposed point values
    unique_point_values = get_unique_values(proposed_points)
    
    # Check whether there are enough proposed point values to map
    if len(unique_point_values) < len(unique_time_ests):
        print('ERROR: There are less unique proposed values than unique '
              'time-estimation values.')
    assert len(unique_point_values) >= len(unique_time_ests)
    
    # Choose a random subset of the proposed point values
    unique_point_values = np.random.choice(proposed_points, replace=False,
                                           size=len(unique_time_ests))
    
    # Sort these values so that they are related to the time estimations
    unique_point_values = sorted(list(unique_point_values))
    if reverse:
        unique_point_values.reverse()
    
    # Map each time_estimation values with the proposed point values
    values_dict = {  # {time_est: point value}
        unique_time_ests[i]: unique_point_values[i]
        for i in range(len(unique_time_ests))
    }
    
    # Return array of point values related to the time-estimation values
    return np.array([values_dict[time_est[i]]
                     for i in range(len(time_est))])


def generate_time_est(node, fn=None, value=None, **dist_params):
    """
    Generator of time-estimation values for a provided list/set of nodes.
    
    Args:
        node: Node whose leaf nodes' time estimation to update.
        fn: Function that generates the time estimations.
        **dist_params: Parameters for the functions/distribution.

    Returns:
        Numpy array of time-estimation values.
    """
    num_nodes = len(node)  # Number of children of the provided node
    
    # Check whether a function or a value has been provided as an argument
    if fn is None and value is None:
        print('ERROR: You have to provide a function or a value!')
    assert fn is not None or value is not None
    
    # Uniform time-estimation values
    if fn is None:
        time_est = np.zeros(num_nodes, dtype=np.int) + value
    
    # Time-estimation values from a function
    else:
        time_est = fn(**dist_params, size=num_nodes)
        
    # Update time estimations
    ch = node.get_ch()
    for i in range(num_nodes):
        ch[i].set_time_est(time_est[i])
    
    return time_est


def get_unique_values(lst):
    """
    Converts a provided list of elements to a sorted list of unique elements.
    
    Args:
        lst: List of elements.

    Returns:
        Sorted list of unique elements.
    """
    return sorted(list(set(lst)))


def update_leaf_node_time_est(node, value):
    """
    Update time-estimation value for the leaf nodes of the provided node.
    
    Args:
        node: Node whose leaf nodes should be updated.
        value: New time-estimation value.

    Returns:
        /
    """
    # If this node is a leaf node, set new parameter value
    if len(node) == 0:
        node.set_time_est(value)

    # Otherwise, find leaf nodes of this node
    else:
        for ch in node.get_ch():
            update_leaf_node_time_est(ch, value)

    return
