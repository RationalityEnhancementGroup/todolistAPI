import itertools
import numpy as np
import random
import time

from collections import deque
from copy import deepcopy
from math import ceil, gcd
from todolistMDP import mdp

POWERS = {
    0: 1.
}


class Item:
    def __init__(self, description, completed=False, deadline=None,
                 deadline_datetime=None, effective_deadline=None, item_id=None,
                 latest_start_time=0, penalty=0, prob=1., reward=0,
                 root_items=None, scheduled=False, sub_items=None,
                 super_items=None, time_est=None):
        """
        TODO: Fill this...
            - For some parameters, we have to check whether all sub-items...
        
        TODO:
            - Implement solve function for an item so that the ToDoListMDP
              can be solved recursively.
            - What parameters have to be passed down?
            - What parameters have to be passed up?
            - Maybe add value as a parameter so that reward expresses immediate
              reward and not ...?
            - Fix effective deadline calculation!
            - More complex scenarios:
                - Sub-items have their own values
            - Completed bottom-up updates
        
        Args:
            description: Description of the item, i.e. item name.
            
            completed: Whether the item have been completed.
            deadline:
            deadline_datetime:
            effective_deadline:
            item_id:
            latest_start_time:
            penalty:
            prob: Probability of successful item completion.
            reward: Reward for executing the task on time.
            root_items: Goals to which this item belongs to.
            scheduled:
            sub_items:
            super_items:
            time_est: Units of time required to execute the item.
        """
        
        self.description = description
        
        self.item_id = item_id
        if self.item_id is None:
            self.item_id = self.description
        
        self.root_items = root_items
        if self.root_items is None:
            self.root_items = set()

        self.sub_items = sub_items
        if self.sub_items is None:
            self.sub_items = deque()  # Children items have to be ordered!

        self.super_items = super_items
        if self.super_items is None:
            self.super_items = set()

        self.completed = completed
        self.deadline = deadline
        self.deadline_datetime = deadline_datetime
        
        # Set effective deadline
        self.effective_deadline = effective_deadline
        if self.effective_deadline is None:
            self.effective_deadline = self.deadline
        
        self.latest_start_time = latest_start_time
        self.penalty = penalty
        self.prob = prob
        self.reward = reward
        self.scheduled = scheduled
        self.time_est = time_est  # TODO: Remove (?!)

        # Initialize attribute for optimal rewards
        self.optimal_reward = None

        # Calculate time estimation
        self.completed_time_est = 0  # Time estimate of completed tasks
        self.scheduled_time_est = 0  # Time estimate of scheduled tasks
        self.unscheduled_time_est = 0  # Time estimate of uncompleted tasks
        self.total_time_est = 0  # Time estimate of all tasks
        
        if time_est is not None:
            self.total_time_est += time_est
            
            if self.completed:
                self.completed_time_est += time_est
            else:
                if self.scheduled:
                    self.scheduled_time_est += time_est
                else:
                    self.unscheduled_time_est += time_est
        
        # Split tasks into completed and uncompleted
        # self.all_sub_items = deque()
        self.completed_sub_items = deque()
        self.scheduled_sub_items = deque()
        self.unscheduled_sub_items = deque()

        # Add item to the queue of tasks
        if sub_items is not None:
            self.add_sub_items(sub_items)
        
        # Calculate points per hour
        self.points_per_hour = 0
        self.compute_points_per_hour()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self.effective_deadline == other.effective_deadline

    def __ne__(self, other):
        return self.effective_deadline != other.effective_deadline

    def __ge__(self, other):
        return self.effective_deadline >= other.effective_deadline

    def __gt__(self, other):
        return self.effective_deadline > other.effective_deadline

    def __le__(self, other):
        return self.effective_deadline <= other.effective_deadline

    def __lt__(self, other):
        return self.effective_deadline < other.effective_deadline

    def __str__(self):
        # TODO: Add other attributes
        #     - Change parents, children etc..
        #     - Number of sub-tasks
        #     - Time estimates
        
        parents_string = "\n".join(item.get_description()
                                   for item in self.super_items
                                   if self.super_items is not None)
        
        return f'Description: {self.description}\n' \
               f'Completed: {self.completed}\n' \
               f'Deadline: {self.deadline}\n' \
               f'Deadline datetime: {self.deadline_datetime}\n' \
               f'Effective deadline: {self.effective_deadline}\n' \
               f'ID: {self.item_id}\n' \
               f'Latest start time: {self.latest_start_time}\n' \
               f'Parents(s): {parents_string}\n' \
               f'Penalty: {self.penalty}\n' \
               f'Points per hour: {self.points_per_hour}\n' \
               f'Probability: {self.prob}\n' \
               f'Reward: {self.reward}\n' \
               f'Scheduled: {self.scheduled}\n' \
               f'Time est.: {self.time_est}\n'

    def _add_completed_task(self, item):
        self.completed_sub_items.append(item)
        self.completed_time_est += item.get_time_est()

    def _add_scheduled_task(self, task):
        self.scheduled_sub_items.append(task)
        self.scheduled_time_est += task.get_time_est()

    def _add_unscheduled_task(self, task):
        self.unscheduled_sub_items.append(task)
        self.unscheduled_time_est += task.get_time_est()
        
    def _add_non_leaf_sub_items(self, item):
        self.completed_sub_items.extend(item.get_completed_sub_items())
        self.completed_time_est += item.get_completed_time_est()
        
        self.scheduled_sub_items.extend(item.get_scheduled_sub_items())
        self.scheduled_time_est += item.get_scheduled_time_est()
        
        self.unscheduled_sub_items.extend(item.get_unscheduled_sub_items())
        self.unscheduled_time_est += item.get_unscheduled_time_est()
        
        self._update_effective_deadline(item)
        self._update_latest_start_time(item)
        
    def _distribute_sub_item(self, sub_item):
        if sub_item.is_completed():
            self._add_completed_task(sub_item)
        else:
            if sub_item.is_scheduled():
                self._add_scheduled_task(sub_item)
            else:
                self._add_unscheduled_task(sub_item)

    def _update_effective_deadline(self, item):
        item_effective_deadline = item.get_effective_deadline()
        if self.effective_deadline is None:
            self.effective_deadline = item_effective_deadline
        if item_effective_deadline is not None:
            self.effective_deadline = min(self.effective_deadline,
                                          item_effective_deadline)
            
    def _update_latest_start_time(self, item):
        item_latest_start_time = item.get_latest_start_time()
        if item_latest_start_time is not None:
            self.latest_start_time = min(self.latest_start_time,
                                         item_latest_start_time)

    # def add_completed_time(self, time_est):
    #     self.completed_time_est += time_est
    #     self.unscheduled_time_est -= time_est
    #     self.update_total_time_est()
    
    def add_root_item(self, root_item):
        self.root_items.add(root_item)

    def add_sub_item(self, sub_item):
        
        # Add reward
        self.reward += sub_item.get_reward()
        
        # If it is a leaf node
        if len(sub_item.get_sub_items()) == 0:
            # self.all_sub_items.append(sub_item)

            self._distribute_sub_item(sub_item)
            self._update_effective_deadline(sub_item)
            self._update_latest_start_time(sub_item)
            
        else:
            self._add_non_leaf_sub_items(sub_item)  # TODO: Maybe recursive call (?)

        # Set item goal
        sub_item.add_super_item(self)  # TODO: Implement this...
    
        # Update total time estimate (of all sub-items)
        self.update_total_time_est()

    def add_sub_items(self, sub_items):
        for sub_item in sub_items:
            self.add_sub_item(sub_item)
            # self.value_est += task.get_prob() * task.get_reward()

    def add_super_item(self, super_item):
        self.super_items.add(super_item)

    def compute_latest_start_time(self):

        if self.get_uncompleted_time_est() > self.deadline:
            raise Exception(self.generate_unattainable_msg())

        # Get all uncompleted sub-items of the current sub-item
        items = list(self.get_uncompleted_sub_items())

        # Sort items w.r.t. effective deadline
        items.sort()

        current_time = self.deadline

        for item in reversed(items):
            current_time = min(current_time, item.get_deadline())
            current_time -= item.get_uncompleted_time_est()

            # Assign latest start time | TODO: Not true. It varies with sorting!
            # item.latest_start_time = current_time

            if current_time < 0:
                raise Exception(item.generate_unattainable_msg())

        self.latest_start_time = current_time
        self.effective_deadline = self.latest_start_time + \
                                  self.get_uncompleted_time_est()

    def compute_points_per_hour(self):
        self.points_per_hour = self.reward / self.total_time_est * 60

    def generate_unattainable_msg(self):
        return f"Item {self.get_description()} is unattainable!"

    def get_all_sub_items(self):
        return self.completed_sub_items + self.get_uncompleted_sub_items()

    def get_completed_sub_items(self):
        return self.completed_sub_items

    def get_completed_time_est(self):
        return self.completed_time_est
    
    def get_copy(self):
        return deepcopy(self)

    def get_deadline(self):
        return self.deadline

    def get_deadline_datetime(self):
        return self.deadline_datetime

    def get_description(self):
        return self.description

    def get_effective_deadline(self):
        return self.effective_deadline

    def get_item_id(self):
        return self.item_id
    
    def get_latest_deadline_time(self):
        # TODO: After implemented multiple deadlines
        raise NotImplementedError()
    
    def get_latest_start_time(self):
        return self.latest_start_time
    
    def get_next_reward(self, time=0):
        #     # If the latest deadline has not been met, get no reward
        #     if time > self.get_latest_deadline_time():
        #         return 0
        #
        #     # Otherwise, get the reward for the next deadline that has been met
        #     times = sorted(self.rewards.keys())
        #     t = next(val for x, val in enumerate(times) if val >= time)
        #
        #     return self.rewards[t]
        raise NotImplementedError()
    
    def get_optimal_reward(self):
        return self.optimal_reward

    def get_penalty(self):
        return self.penalty
    
    def get_points_per_hour(self):
        return self.points_per_hour

    def get_prob(self):
        return self.prob

    def get_reward(self):
        return self.reward
    
    def get_reward_dict(self):
        raise NotImplementedError()

    def get_root_items(self):
        return self.root_items
    
    def get_scheduled_sub_items(self):
        return self.scheduled_sub_items

    def get_scheduled_time_est(self):
        return self.scheduled_time_est

    def get_sub_items(self):
        return self.sub_items

    def get_super_items(self):
        return self.super_items

    def get_time_est(self):  # TODO: Return total time estimate (?!)
        return self.time_est
    
    def get_total_time_est(self):
        return self.total_time_est

    def get_uncompleted_sub_items(self):
        return self.scheduled_sub_items + self.unscheduled_sub_items

    def get_uncompleted_time_est(self):
        return self.scheduled_time_est + self.unscheduled_time_est

    def get_unscheduled_sub_items(self):
        return self.unscheduled_sub_items

    def get_unscheduled_time_est(self):
        return self.unscheduled_time_est
    
    def init_sub_trees(self, _check_root=True):
        if _check_root and len(self.super_items) > 0:
            raise Exception("Item is not a root node!")
        
        for item in self.sub_items:
            
            # Assign reward if no reward was assigned
            # if item.get_reward() == 0:
            #     frac_time = item.get_total_time_est() / self.total_time_est
            #     reward = frac_time * self.reward
            #     item.set_reward(reward)
                
            # Assign deadline if no deadline was assigned
            if item.get_deadline() is None:
                item.set_deadline(self.deadline)
                
            # Recursive call to initialize sub items
            item.init_sub_trees(_check_root=False)

    def is_completed(self, check_sub_items=False):
        """
        Method for checking whether a goal is complete by checking
        if all tasks are complete

        Args:
            check_sub_items: Whether to check whether all tasks are completed
                         or to return the cached value

        Returns:
            Completion status
        """
        if check_sub_items:
            for task in self.sub_items:
                if not task.is_completed():
                    return False
                
            # If all sub-tasks are completed
            self.completed = True
            
        return self.completed

    def is_scheduled(self):
        return self.scheduled
    
    def print(self, indent=2, factor=0):
        print(f"{' '* indent * factor}> {self.description}")
        for item in self.sub_items:
            item.print(indent=indent, factor=factor+1)

    def remove_item(self, item):
        raise NotImplementedError()
        # if item in self.all_tasks:
        #     self.all_tasks.remove(item)
        #     task_time_est = item.get_time_est()
        #
        #     # Subtract task time estimation
        #     if item.is_completed():
        #         self.completed_time_est -= task_time_est
        #     else:
        #         self.uncompleted_time_est -= task_time_est
        #
        #     self.total_time_est -= task_time_est
        #
        #     # Subtract task value
        #     self.value_est -= item.get_prob() * item.get_reward()

    def reset_completed(self):
        # self.completed = False
        # self.uncompleted_time_est = 0
        # self.completed_time_est = 0
        #
        # for task in self.all_tasks:
        #     task.set_completed(False)
        #     self.uncompleted_time_est += task.get_time_est()
        #
        # self.update_total_time_est()
        raise NotImplementedError()
    
    @staticmethod
    def scale_parameter(param, scale, rounding=False, up=True):
        # TODO: Make this function for all "scalable" parameters
        if scale != 1:
            if up:
                param *= scale
                if rounding:
                    param = ceil(param)
            else:
                if rounding:
                    param = ceil(param / scale)
                else:
                    if gcd(scale, param) == 1:
                        raise ArithmeticError(
                            f"{param.__name__} with value {param} "
                            f"cannot be divided by {scale}!")
                    param //= scale
        return param
    
    # def scale_est_deadline(self, scale, up=True):
    #     if up:
    #         self.effective_deadline *= scale
    #     else:
    #         self.effective_deadline //= scale
    #
    # def scale_latest_start_time(self, scale, up=True):
    #     if up:
    #         self.latest_start_time *= scale
    #     else:
    #         self.latest_start_time //= scale
    #
    # def scale_unscheduled_time_est(self, scale, up=True):
    #     if up:
    #         self.unscheduled_time_est *= scale
    #     else:
    #         self.unscheduled_time_est //= scale

    def set_completed(self, completed: bool):
        # Task...
        # if self.completed != completed:
        #     self.completed = completed
        #     if completed:
        #         self.goal.add_completed_time(self.time_est)
        #         # TODO: Check whether this is the last completed task
        #     else:
        #         self.goal.add_completed_time(-self.time_est)
        #         # TODO: Check whether this makes a completed goal active again

        # Goal...
        #     self.completed = completed
        #     if completed:
        #         for task in self.all_tasks:
        #             task.set_completed(completed)
        #
        #         # Change time-estimation values
        #         self.completed_time_est = self.total_time_est
        #         self.uncompleted_time_est = 0
        #         self.update_total_time_est()

        raise NotImplementedError()

    def set_deadline(self, value):
        self.deadline = value

    def set_optimal_reward(self, value):
        self.optimal_reward = value

    def set_reward(self, value):
        self.reward = value
        self.compute_points_per_hour()

    def set_rewards_dict(self, rewards: dict):
        # self.rewards = rewards
        # self.latest_deadline_time = max(rewards.keys())
        raise NotImplementedError()
    
    def solve(self, gamma=1., loss_rate=1.):
        # TODO: Extend this to sub-goals (?!)
        
        # items = list(self.get_uncompleted_sub_items())
        time_est = self.get_uncompleted_time_est()
        
        est_loss = time_est * (- loss_rate) * np.power(gamma, time_est)
        print(self.description, time_est, est_loss)
        
        # items.sort()
        
        for item in self.sub_items:
            item.solve(gamma=gamma, loss_rate=loss_rate)
        
        # TODO:
        # TODO: Optimal policy is actually a list of tasks
        
        # if len(self.super_items) != 0:
        #     raise Exception(f"{self.description} is not a goal item.")
        
        # raise NotImplementedError()
        
    def update_info_recursively(self):
        """
        TODO:
        - Completed tasks
        - Scheduled tasks
        - Uncompleted tasks

        - Completed time est
        - Scheduled time est
        - Uncompleted time est

        - Optimal reward
        - Probability of success
        """
        raise NotImplementedError()

    def update_total_time_est(self):
        self.total_time_est = self.completed_time_est + \
                              self.scheduled_time_est + \
                              self.unscheduled_time_est


class ToDoListMDP(mdp.MarkovDecisionProcess):
    """
    State: (boolean vector for task completion, time)
    """

    def __init__(self, to_do_list: Item, end_time=None, gamma=1.0,
                 generate_state_space=False, living_reward=0.0, loss_rate=0.0,
                 noise=0.0, start_time=0):
        """
        
        Args:
            to_do_list: root Item object (consists of all goals)
            
            end_time:  # End time of the MDP (i.e. horizon)
            gamma: Discount factor
            living_reward: (Negative) reward for exiting "normal" states.
            loss_rate: Loss rate (lambda in the report).
            noise: Probability of moving in an unintended direction.
            start_time:  # Starting time of the MDP
        """
        # Initialize powers
        POWERS[1] = gamma
        
        # To-do list
        self.to_do_list = to_do_list
        
        # Initialize times
        self.start_time = start_time
        self.current_time = self.start_time

        # Set MDP horizon
        self.end_time = to_do_list.get_effective_deadline() + 1
        if end_time is not None:
            
            # Make sure that the proposed end_time is not out of bounds!
            self.end_time = min(self.end_time, end_time)
            
        # Initialize other parameters
        self.gamma = gamma
        self.living_reward = living_reward
        self.loss_rate = loss_rate
        self.noise = noise

        # Get a complete list of goals
        self.goals = self.to_do_list.get_sub_items()
        
        # Initialize set of completed and uncompleted task indices
        self.completed_task_idx = set()
        self.uncompleted_task_idx = set()

        # TODO: Check whether task descriptions are valid
        # Create set of task indices for each goal according to the MDP indexing
        # Create mapping from indices to tasks represented as list
        # Create mapping from tasks to indices represented as dict
        self.goal_tasks_to_indices = dict()
        self.index_to_task = deque()
        self.task_to_index = dict()

        for goal in self.goals:
            
            # Initialize set of task indices for a goal
            self.goal_tasks_to_indices[goal] = set()
            
            for task in goal.get_uncompleted_sub_items():
                
                # Add task to the list of all uncompleted tasks
                self.index_to_task.append(task)
            
                # Add task index to goal's set of task indices
                self.goal_tasks_to_indices[goal].add(self.task_to_index[task])
                
                # Create task to index mapping
                idx = len(self.index_to_task) - 1
                self.task_to_index[task] = idx
                
                # TODO: Potentially unnecessary?!
                if task.is_completed():
                    self.completed_task_idx.add(idx)
                else:
                    self.uncompleted_task_idx.add(idx)

                # Make connection from task to goal
                task.add_root_item(goal)

        # Initialize start state
        self.start_state = self.get_start_state()
        
        # Generate state space
        self.states = deque()
        if generate_state_space:
            self.states.extend(self.generate_complete_state_space())
        else:
            self.states.append(self.start_state)

        # Mapping from (binary vector, time) to integer (?!)
        # TODO: Potentially unnecessary...
        # self.state_to_index = {self.states[i]: i
        #                        for i in range(len(self.states))}

        # State values | {state: value}
        self.v_states = {
            self.start_state: np.NINF
        }
        
        # Optimal policy | {state: action}
        self.optimal_policy = {
            self.start_state: None
        }
        
        # Pseudo-rewards (PR) | {(s, a, s'): PR(s, a, s')}
        self.pseudo_rewards = {
            self.start_state: None
        }
        
        # Transformed pseudo-rewards (TPR) | {(s, a, s') --> PR'(s, a, s')}
        self.transformed_pseudo_rewards = {
            self.start_state: None
        }
        
        # Create DAG
        # self.reverse_DAG = MDPGraph(self)
        # self.linearized_states = self.reverse_DAG.linearize()

        # Calculate PRs for each state
        # self.calculate_pseudo_rewards()

        # Apply linear transformation to PR'
        # self.transform_pseudo_rewards()
        
    def calculate_power(self, power):
        if power in POWERS.keys():
            return POWERS[power]
        
        value = self.calculate_power(power // 2) ** 2
        
        # If there is remainder
        if power % 2 == 1:
            value *= POWERS[1]
            
        # Store calculated value
        POWERS[power] = value
        
        return value

    def generate_complete_state_space(self):
        
        # Calculate the total number of tasks in the to-do list
        num_tasks = len(self.to_do_list.get_uncompleted_sub_items())

        # Initialize linked list
        states = deque()
        
        # Generate complete state space
        for t in range(self.end_time):  #  + 2):
            for bit_vector in itertools.product([0, 1], repeat=num_tasks):
                state = (bit_vector, t)
                states.append(state)
                
                self.v_states[state] = np.NINF
                self.optimal_policy[state] = None
                self.pseudo_rewards[state] = None
                self.transformed_pseudo_rewards[state] = None
                
        return states
        
    def get_current_time(self):
        return self.current_time

    def get_end_time(self):
        return self.end_time

    def get_gamma(self):
        return self.gamma

    def get_optimal_policy(self, state=None):
        """
        Returns the mapping of state to the optimal policy at that state
        """
        if state is not None:
            return self.optimal_policy[state]
        return self.optimal_policy

    def get_pseudo_rewards(self, transformed=False):
        """ TODO: Check whether the description still holds...
        pseudo_rewards is stored as a dictionary,
        where keys are tuples (s, s') and values are PR'(s, a, s')
        """
        if transformed:
            return self.transformed_pseudo_rewards
        return self.pseudo_rewards

    def get_start_time(self):
        return self.start_time

    def get_state_value(self, state):
        return self.v_states[state][0]

    def get_states(self):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        return self.states

    def get_tasks_list(self):
        return self.index_to_task

    def increase_time(self, value=1):
        self.current_time += value

    def set_living_reward(self, living_reward):
        """
        The (negative) reward for exiting "normal" states. Note that in the R+N
        text, this reward is on entering a state and therefore is not clearly
        part of the state's future rewards.
        """
        self.living_reward = living_reward

    def set_noise(self, noise):
        """
        Sets the probability of moving in an unintended direction.
        """
        self.noise = noise

    def solve(self):
        state, t = self.start_state
        
        for idx in state:
            task = self.index_to_task[idx]
            
        self.to_do_list.solve(gamma=self.gamma, loss_rate=self.loss_rate)
        
        # for goal in self.goals:
        #     goal.solve()
        # raise NotImplementedError()

    @staticmethod
    def tasks_to_binary(tasks):
        """
        Convert a list of Item objects to a bit vector with 1 being complete and
        0 if not complete.
        """
        return [1 if task.is_completed() else 0 for task in tasks]
    
    # ========================= ToDoList functions =========================
    def action(self, loss_rate=1., task=None):  # TODO: Modify (!)
        """
        Do a specified action

        Args:
            loss_rate: TODO
            task: ...; If not defined, it completes a random task

        Returns:

        """
        # Randomly get an uncompleted task from an uncompleted goal
        while task is None:
            idx = random.sample(self.uncompleted_task_idx, 1)[0]
            
            task = self.index_to_task[idx]
            if task.is_completed():
                task = None

        time_est = task.get_time_est()
        
        # curr_time = self.current_time
        next_time = self.current_time + time_est
        
        # curr_power = self.calculate_power(power=curr_time)
        next_power = self.calculate_power(power=next_time)
        # actl_power = next_power - curr_power
        
        # Initialize task reward (immediate loss for working on a task)
        reward = next_power * (- loss_rate) * time_est

        # Move time_est steps in the future
        self.increase_time(task.get_time_est())

        # Get goal reward if succesful task completing completes goal before
        # its deadline
        reward += self.do_task(task)

        # TODO: Check whether some deadlines have been surpassed and get penalty
        # reward += self.check_deadlines(self.current_time, next_time)

        return reward

    def check_deadlines(self, prev_time, curr_time):  # TODO: Modify (!)
        """
        Check which goals passed their deadline between prev_time and curr_time
        If goal passed deadline during task, incur penalty
        """
        # penalty = 0
        #
        # for goal in self.uncompleted_goals:
        #     # Check:
        #     # 1) goal is now passed deadline at curr_time
        #     # 2) goal was not passed deadline at prev_time
        #
        #     # TODO: Shouldn't we have an inequality in one of the tests?!
        #     if curr_time > goal.get_latest_deadline_time() and \
        #             not prev_time > goal.get_latest_deadline_time():
        #         penalty += goal.get_penalty()
        #
        # return penalty
        raise NotImplementedError()
    
    def check_goal_completion(self, goal):
        # If completion of the task completes the goal
        for task_idx in self.goal_tasks_to_indices[goal]:
            # TODO: How to access the state vector at current time?
            pass
        raise NotImplementedError()

    def do_task(self, task):  # TODO: Modify (!)
        # TODO: Change this so that it is on a Goal level (!)
        # Get probability of success
        threshold = task.get_prob()

        # Generate a random number from Uniform([0, 1])
        p = random.uniform()
        
        # Initialize total reward (obtained by potential goal completion)
        total_reward = 0
    
        # Check whether the task is completed on time
        if p < threshold and self.current_time <= task.get_deadline():
            
            # For all goals to which the task belongs
            for goal in task.get_root_items():
                if goal.is_completed():
            
                    # TODO: Procedure to implement
                    #     - Get task's goal
                    #     - Check whether the task completes the goal
                    #     - If yes, return goal's reward
                    
                    task.set_completed(True)
                    self.unscheduled_tasks.remove(task)
                    self.completed_tasks.append(task)
                
                    if goal.is_completed():
                        self.uncompleted_goals.remove(goal)
                        self.completed_goals.append(goal)
    
                        # Add goal completion reward
                        total_reward += goal.get_reward(self.current_time)
    
        return total_reward

    # ========================== OLD FUNCTIONS ============================
    
    def calculate_optimal_values_and_policy(self):
        """
        Given a ToDoListMDP, perform value iteration/backward induction to find
        the optimal value function

        Input: ToDoListMDP
        Output: Dictionary of optimal value of each state
        """
        self.optimal_policy = {}  # state --> action
        self.v_states = {}  # state --> (value, action)
    
        # Perform Backward Iteration (Value Iteration 1 Time)
        for state in self.linearized_states:
            self.v_states[state], self.optimal_policy[state] = \
                self.get_value_and_action(state)

    def calculate_pseudo_rewards(self):
        """
        private method for calculating untransformed pseudo-rewards PR
        """
        for state in self.states:
            for action in self.get_possible_actions(state):
                for next_state, prob in \
                        self.get_trans_states_and_probs(state, action):
                    reward = self.get_reward(state, action, next_state)
                    pr = self.v_states[next_state] - \
                         self.v_states[state] + reward
                    self.pseudo_rewards[(state, action, next_state)] = pr

    def transform_pseudo_rewards(self, print_values=False):
        """
        TODO: Understand what the method REALLY does...
        
        applies linear transformation to PRs to PR'

        linearly transforms PR to PR' such that:
            - PR' > 0 for all optimal actions
            - PR' <= 0 for all suboptimal actions
        """
        # Calculate the 2 highest pseudo-rewards
        highest = -float('inf')
        sec_highest = -float('inf')
    
        for trans in self.pseudo_rewards:
            pr = self.pseudo_rewards[trans]
            if pr > highest:
                sec_highest = highest
                highest = pr
            elif sec_highest < pr < highest:
                sec_highest = pr

        # TODO: Understand this...
        alpha = (highest + sec_highest) / 2
        beta = 1
        if alpha <= 1.0:
            beta = 10

        # TODO: Why (alpha + pr) * beta?! Shouldn't it be (alpha + pr * beta)!?
        for trans in self.pseudo_rewards:
            self.transformed_pseudo_rewards[trans] = \
                (alpha + self.pseudo_rewards[trans]) * beta
            
        if print_values:
            print(f'1st highest: {highest}')
            print(f'2nd highest: {sec_highest}')
            print(f'Alpha: {alpha}')

    def scale_rewards(self, min_value=1, max_value=100, print_values=False):
        """
        Linear transform we might want to use with Complice
        """
        dict_values = np.asarray([*self.pseudo_rewards.values()])
        minimum = np.min(dict_values)
        ptp = np.ptp(dict_values)
        for trans in self.pseudo_rewards:
            self.transformed_pseudo_rewards[trans] = \
                max_value * (self.pseudo_rewards[trans] - minimum)/(ptp)

    # ===== Getters =====
    def get_expected_pseudo_rewards(self, state, action, transformed=False):
        """
        Return the expected pseudo-rewards of a (state, action) pair
        """
        expected_pr = 0.0
        for next_state, prob in self.get_trans_states_and_probs(state, action):
            if transformed:
                expected_pr += \
                    prob * self.transformed_pseudo_rewards[(state, action,
                                                            next_state)]
            else:
                expected_pr += prob * self.pseudo_rewards[(state, action,
                                                           next_state)]
        return expected_pr

    def get_linearized_states(self):
        return self.linearized_states

    def get_possible_actions(self, state):
        """
        Return list of possible actions from 'state'.
        Returns a list of indices
        """
        # TODO: Think of a better implementation that does not create a new
        #        list on every call on this function.
        tasks = state[0]
        
        if not self.is_terminal(state):
            actions = [i for i, task in enumerate(tasks) if task == 0]
            return actions
        
        return []  # Terminal state --> No actions

    def get_q_value(self, state, action):
        """

        Args:
            state: current state (tasks, time)
            action: index of action in MDP's tasks

        Returns:
            Q-value of state
        """
        q_value = 0

        for next_state, prob in self.get_trans_states_and_probs(state, action):
            next_state_value = self.v_states[next_state]

            q_value += prob * (self.get_reward(state, action, next_state) +
                               self.gamma * next_state_value)

        return q_value

    def get_reward(self, state, action, next_state):
        """
        Get the reward for the state, action, next_state transition.
        state: (list of tasks, time)
        action: integer of index of task
        next_state: (list of tasks, time)

        Not available in reinforcement learning.
        """
        reward = 0
        task = self.index_to_task[action]
        goal = task.get_super_items()
        prev_tasks, prev_time = state
        next_tasks, next_time = next_state
    
        # Get reward for doing a task
        reward += task.get_reward()
    
        # TODO: Simpler computation of rewards & penalties
        # Reward for goal completion
        if next_tasks[action] == 1:
            if self.is_goal_completed(goal, next_state) and \
                    self.is_goal_active(goal, next_time):
                reward += goal.get_reward(next_time)
    
        # Penalty for missing a deadline
        # TODO: You cannot implement "lazy check" because in that way you would
        #        not know whether you have missed a deadline or not?!
        for goal in self.goals:
            if not self.is_goal_completed(goal, state) and \
                    self.is_goal_active(goal, prev_time) and not \
                    self.is_goal_active(goal, next_time):
                # If a deadline passed during time of action, add penalty
                reward += goal.get_penalty()
    
        return reward

    def get_start_state(self):
        """
        Return the start state of the MDP.
        """
        start_state =\
            self.tasks_to_binary(self.to_do_list.get_uncompleted_sub_items())
        return start_state, self.start_time

    # def get_state_index(self, state):
    #     TODO: Potentially unnecessary function...
    #     return self.state_to_index[state]

    def get_trans_states_and_probs(self, state, action=None):
        """
        Returns list of (next_state, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.

        Note that in Q-Learning and reinforcement
        learning in general, we do not know these
        probabilities nor do we directly model them.
        """
        next_states_probs = []

        # If there is no available action
        if action is None:
            return next_states_probs  # Empty list / Terminal state
        
        # Action is the index that is passed in
        task = self.index_to_task[action]
        binary_tasks = list(state[0])[:]
        new_time = state[1] + task.get_time_est()

        # Extend the end time of the to-do list (latest deadline time)
        if new_time > self.to_do_list.get_end_time():
            new_time = self.to_do_list.get_end_time() + 1

        # Generate all future states if there is a probability that the action
        # might not be completed successfully
        tasks_no_completion = binary_tasks[:]
        if 1 - task.get_prob() > 0:  # 1 - P(completion)
            next_states_probs.append(((tuple(tasks_no_completion), new_time),
                                      1 - task.get_prob()))
            
        # Generate all future states if there is no probability that the action
        # might not be completed successfully
        tasks_completion = binary_tasks[:]
        tasks_completion[action] = 1
        if task.get_prob() > 0:
            next_states_probs.append(((tuple(tasks_completion), new_time),
                                      task.get_prob()))

        return next_states_probs

    def get_value_and_action(self, state):
        """
        Input:
        mdp: ToDoList MDP
        state: current state (tasks, time)
        V_states: dictionary mapping states to current best (value, action)

        Output:
        best_value: value of state state
        best_action: index of action that yields highest value at current state
        """
        possible_actions = self.get_possible_actions(state)
        best_action = None
        best_value = -float('inf')

        if self.is_terminal(state):
            best_value = 0
            best_action = None  # There is no action that can be performed!
            return best_value, best_action

        for action in possible_actions:
            q_value = self.get_q_value(state, action)
            if best_value < q_value:
                best_value = q_value
                best_action = action

        return best_value, best_action

    def get_value_function(self):
        """
        To get the state value for a given state, use get_state_value(state)
        
        Returns:

        """
        return self.v_states

    def is_goal_active(self, goal, time):
        """
        Given a Goal object and a time
        Check if the goal is still active at that time
        Note: completed goal is still considered active if time has not passed
              the deadline
        """
        return time <= goal.get_latest_deadline_time() and \
               time <= self.to_do_list.get_end_time()

    def is_goal_completed(self, goal, state):
        """
        Given a Goal object and current state
        Check if the goal is completed
        """
        tasks = state[0]
    
        for i in self.goal_tasks_to_indices[goal]:
            if tasks[i] == 0:
                return False
    
        return True

    def is_task_active(self, task, time):
        """
        Check if the goal for a given task is still active at a time
        """
        goal = task.get_super_items()
        return self.is_goal_active(goal, time)

    def is_terminal(self, state):
        """
        Returns true if the current state is a terminal state.  By convention,
        a terminal state has zero future rewards.  Sometimes the terminal
        state(s) may have no possible actions.  It is also common to think of
        the terminal state as having a self-loop action 'pass' with zero reward;
        the formulations are equivalent.
        """
        tasks, time = state
        
        # Check if the global end time is reached or if all tasks are completed
        if time > self.to_do_list.get_end_time() or 0 not in tasks:
            return True
        
        # Check if there are any goals that are still active and not completed
        for goal in self.goals:
            if self.is_goal_active(goal, time) and not \
                    self.is_goal_completed(goal, state):
                return False
        
        # TODO: Are there any other conditions to check whether a state is
        #        terminal or not?
        return True


class MDPGraph:
    """
    
    """
    def __init__(self, mdp):
        print('Building reverse graph...')
        
        start = time.time()
        
        self.edges = {}  # state --> {edges}
        self.mdp = mdp
        self.pre_order = {}
        self.post_order = {}
        self.vertices = set()

        # Initialize variables
        self.counter = None
        self.linearized_states = None
        
        # Connecting the graph in reverse manner | next_state --> curr_state
        for state in mdp.get_states():
            self.vertices.add(state)
            self.edges.setdefault(state, set())
            
            for action in mdp.get_possible_actions(state):
                for next_state, prob in \
                        mdp.get_trans_states_and_probs(state, action):
                    self.edges.setdefault(next_state, set())
                    self.edges[next_state].add(state)  # next_state --> state
                    
        print('Done!')
        
        end = time.time()
        print(f'Time elapsed: {end - start:.4f} seconds.\n')
        
    def dfs(self):
        visited_states = {}
        self.counter = 1

        def explore(v):
            """
            
            Args:
                v: vertex

            Returns:

            """
            visited_states[v] = True
        
            # Enter/In time
            self.pre_order[v] = self.counter
            self.counter += 1
            
            for u in self.edges[v]:
                if not visited_states[u]:
                    explore(u)
                    
            # Exit/Out time
            self.post_order[v] = self.counter
            self.counter += 1

        for v in self.vertices:
            visited_states[v] = False

        for v in self.vertices:
            if not visited_states[v]:
                explore(v)

    def linearize(self):
        """
        Returns list of states in topological order
        """
        self.dfs()  # Run DFS to get post_order
        # post_order_dict = self.dfs(self.reverse_graph)
        arr = np.array([(v, -self.post_order[v]) for v in self.post_order],
                       dtype=[('state', tuple), ('post_order', int)])
        reverse_post_order = np.sort(arr, order='post_order')

        self.linearized_states = [state for (state, i) in reverse_post_order]
        
        return self.linearized_states

    # ===== Getters =====
    def get_vertices(self):
        return self.vertices

    def get_edges(self):
        return self.edges


# class ToDoList:
#     def __init__(self, goals):
#         """
#
#         Args:
#             goals: List of all goals
#         """
#         # Initialize goal lists
#         self.all_goals = deque()
#
#         # Initialize task lists
#         self.all_tasks = deque()
#         self.completed_tasks = deque()
#         self.scheduled_tasks = deque()
#         self.unscheduled_tasks = deque()
#
#         # Set latest effective deadline to bound the time horizon
#         self.latest_deadline = float('-inf')
#         self.effective_deadline = float('-inf')
#
#         # Add goals and tasks to the to-do list
#         self.add_goals(goals)
#
#     def __str__(self):
#         return f'Completed Goals: {str(self.all_goals)}\n' \
#                f'Effective deadline: {self.effective_deadline}\n'
#                # f'Latest deadline: {self.latest_deadline}\n'
#
#     def get_effective_deadline(self, update=False):
#         # TODO: Recursive update
#         if update:
#             self.update_effective_deadline()
#         return self.effective_deadline
#
#     def get_goals(self, goal_idx=None):
#         # If no indices specified, return all goals
#         if goal_idx is None:
#             return self.all_goals
#
#         # If there is 1 specified index
#         if type(goal_idx) is int:
#             return self.all_goals[goal_idx]
#
#         # If there are multiple specified indices in a collection
#         return [self.all_goals[idx] for idx in goal_idx]
#
#     def get_latest_deadline(self, update=False):
#         # TODO: Recursive update
#         if update:
#             self.update_latest_deadline()
#         return self.latest_deadline
#
#     def get_tasks(self, task_idx=None):
#         # If no indices specified, return all tasks
#         if task_idx is None:
#             return self.all_tasks
#
#         # If there is 1 specified index
#         if type(task_idx) is int:
#             return self.all_tasks[task_idx]
#
#         # If there are multiple specified indices in a collection
#         return [self.all_tasks[idx] for idx in task_idx]
#
#     def update_effective_deadline(self):
#         # Initialize latest effective deadline
#         self.effective_deadline = float('-inf')
#
#         # Find the latest effective deadline
#         for goal in self.all_goals:
#             self.effective_deadline = \
#                 max(goal.get_effective_deadline(),
#                     self.effective_deadline)
#
#     def update_latest_deadline(self):
#         # Initialize latest effective deadline
#         self.latest_deadline = float('-inf')
#
#         # Find the latest effective deadline
#         for goal in self.all_goals:
#             self.latest_deadline = \
#                 max(self.latest_deadline,
#                     goal.get_latest_deadline_time())
#
#     # ===== Setters =====
#     def add_goals(self, goals):
#         if issubclass(goals, Item):
#             self.all_goals.append(goals)
#             # self.all_tasks.extend(goals.get_all_tasks())
#
#         else:
#             for goal in goals:
#                 self.all_goals.append(goal)
#
#                 self.add_tasks(goal.get_all_tasks())
#
#                 # Get all tasks
#                 # self.all_tasks.extend(goal.get_all_tasks())
#
#                 # Distribute goal tasks to their corresponding categories
#                 # self.completed_tasks.extend(goal.get_completed_tasks())
#                 # self.scheduled_tasks.extend(goal.get_scheduled_tasks())
#                 # self.unscheduled_tasks.extend(goal.get_unscheduled_tasks())
#
#                 # Update latest effective deadline
#                 self.effective_deadline = \
#                     max(self.effective_deadline,
#                         goal.get_effective_deadline())
#
#                 # Update latest deadline
#                 self.latest_deadline = max(self.latest_deadline,
#                                            goal.get_latest_deadline_time())
#
#     # TODO: Maybe remove all operations on Task?
#     def add_tasks(self, tasks):
#         if type(tasks) is Item:
#             self.all_tasks.append(tasks)
#
#         else:
#             for task in tasks:
#                 self.all_tasks.append(task)
#
#                 # Split tasks into completed and uncompleted
#                 if task.is_completed():
#                     self.completed_tasks.append(task)
#                 else:
#                     # Split uncompleted tasks into scheduled and unscheduled
#                     if task.is_scheduled_today():
#                         self.scheduled_tasks.append(task)
#                     else:
#                         self.unscheduled_tasks.append(task)
#
#         # Check whether all tasks have been properly distributed
#         assert len(self.all_tasks) == len(self.completed_tasks) + \
#                len(self.scheduled_tasks) + len(self.unscheduled_tasks)

