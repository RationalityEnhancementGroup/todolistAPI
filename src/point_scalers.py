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


def utility_scaling(task_list, scale_type="min_max",
                    scale_min=None, scale_max=None):
    min_value = float("inf")
    max_value = -float("inf")
    
    mean_reward = []
    
    for task in task_list:
        task_goal = task.get_goal()
        
        task_time_est = task.get_time_est()
        goal_reward = task_goal.get_reward(0)
        goal_time_est = task_goal.get_uncompleted_time_est()
        
        # Calculate task utility according to its goal value
        task_reward = (goal_reward / goal_time_est) * task_time_est
        task.set_reward(task_reward)
        
        mean_reward += [task_reward]

        # Update minimum and maximum values
        min_value = min(task_reward, min_value)
        max_value = max(task_reward, max_value)
        
    mean_reward = np.mean(mean_reward)

    if scale_min is None:
        scale_min = min_value
    if scale_max is None:
        scale_max = max_value

    for task in task_list:
        task_reward = task.get_reward()
        if scale_type == "min_max":
            task_reward = (task_reward - min_value) / (max_value - min_value) \
                          * (scale_max - scale_min) + scale_min
        elif scale_type == "mean_value":
            task_reward = (task_reward - mean_reward) / (max_value - min_value) \
                          * (scale_max - scale_min) \
                          + ((scale_max - scale_min) / 2)
        else:
            raise Exception("Scaling method not implemented!")
        task.set_reward(task_reward)

