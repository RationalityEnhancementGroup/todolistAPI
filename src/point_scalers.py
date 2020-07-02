import numpy as np

from collections import deque


# def scale_rewards(task_list, min_value=1, max_value=100, print_values=False):
#     """
#     Linear transform we might want to use with Complice
#
#     Args:
#         task_list: [Task]
#         min_value:
#         max_value:
#         print_values:
#
#     Returns:
#
#     """
#     dict_values = np.asarray([*self.pseudo_rewards.values()])
#     minimum = np.min(dict_values)
#     ptp = np.ptp(dict_values)
#     for trans in self.pseudo_rewards:
#         self.transformed_pseudo_rewards[trans] = \
#             max_value * (self.pseudo_rewards[trans] - minimum) / (ptp)


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


# def utility_scaling(task_list, scale_type="no_scaling",
#                     scale_min=None, scale_max=None):
#     # TODO: Not working due to non-existing methods...
#
#     min_value = float("inf")
#     max_value = -float("inf")
#
#     mean_reward = []
#
#     for task in task_list:
#         task_goal = task.get_super_items()
#
#         task_time_est = task.get_time_est()
#         goal_reward = task_goal.get_reward(0)
#         goal_time_est = task_goal.get_total_time_est()
#
#         # Calculate task utility according to its goal value
#         task_reward = (goal_reward / goal_time_est) * task_time_est
#         task.set_reward(task_reward)
#
#         mean_reward += [task_reward]
#
#         # Update minimum and maximum values
#         min_value = min(task_reward, min_value)
#         max_value = max(task_reward, max_value)
#
#     mean_reward = np.mean(mean_reward)
#
#     if scale_min is None:
#         scale_min = min_value
#     if scale_max is None:
#         scale_max = max_value
#
#     if scale_type == "no_scaling":
#         pass
#     else:
#         for task in task_list:
#             if min_value == max_value:
#                 task.set_reward((scale_max + scale_min) / 2)
#             else:
#                 task_reward = task.get_reward()
#                 if scale_type == "min_max":
#                     task_reward = (task_reward - min_value) / (max_value - min_value) \
#                                   * (scale_max - scale_min) + scale_min
#                 elif scale_type == "mean_value":
#                     task_reward = (task_reward - mean_reward) / (max_value - min_value) \
#                                   * (scale_max - scale_min) / 2 \
#                                   + ((scale_max + scale_min) / 2)
#                 else:
#                     raise Exception("Scaling method not implemented!")
#                 task.set_reward(task_reward)
