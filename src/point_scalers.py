import numpy as np

from collections import deque


def scale_optimal_rewards(task_list, scale_min=None, scale_max=None,
                          scale_type='no_scaling'):
    """
    
    Args:
        task_list:
        scale_min:
        scale_max:
        scale_type:
            - no_scaling: Default
            - min_max:
            - mean_value

    Returns:

    """
    if scale_min is None and scale_max is None:
        return task_list

    min_value = float("inf")
    max_value = -float("inf")
    
    mean_reward = deque()

    for task in task_list:
        
        # Get task reward
        task_reward = task.get_optimal_reward()

        # Update minimum and maximum values
        min_value = min(task_reward, min_value)
        max_value = max(task_reward, max_value)

        # Add task reward to the list of all rewards
        mean_reward.append(task_reward)
        
    # Compute mean reward
    mean_reward = np.mean(mean_reward)

    # Initialize interval of viable values
    if scale_min is None:
        scale_min = min_value
    if scale_max is None:
        scale_max = max_value
        
    if scale_type == "no_scaling":
        pass
    
    else:
        
        for task in task_list:
            
            if min_value == max_value:
                task.set_reward((scale_max + scale_min) / 2)
                
            else:
                task_reward = task.get_optimal_reward()
                
                if scale_type == "min_max":
                    task_reward = (task_reward - min_value) / (max_value - min_value) \
                                  * (scale_max - scale_min) + scale_min
                    
                elif scale_type == "mean_value":
                    task_reward = (task_reward - mean_reward) / (max_value - min_value) \
                                  * (scale_max - scale_min) / 2 \
                                  + ((scale_max + scale_min) / 2)
                    
                else:
                    raise Exception("Scaling method not implemented!")
                
                # Set scaled optimal task reward
                task.set_optimal_reward(task_reward)
