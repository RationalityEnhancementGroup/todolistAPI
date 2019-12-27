import numpy as np


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


def utility_scaling(task_list, scale_min=None, scale_max=None):
    min_value = float("inf")
    max_value = -float("inf")
    
    for task in task_list:
        task_goal = task.get_goal()
        
        task_time_est = task.get_time_est()
        goal_reward = task_goal.get_reward(0)
        goal_time_est = task_goal.get_total_time_est()
        
        task_reward = (task_time_est / goal_time_est) * goal_reward
        task.set_reward(task_reward)

        min_value = min(task_reward, min_value)
        max_value = max(task_reward, max_value)

    if scale_min is None:
        scale_min = min_value
    if scale_max is None:
        scale_max = max_value

    for task in task_list:
        task_reward = task.get_reward()
        task_reward = (task_reward - min_value) / (max_value - min_value) * \
                      (scale_max - scale_min) + scale_min
        task.set_reward(task_reward)
