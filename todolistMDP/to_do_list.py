import numpy as np
import time

from collections import deque
from copy import deepcopy
from todolistMDP.zero_trunc_poisson import get_binned_dist
from pprint import pprint


class Item:
    def __init__(self, description, completed=False, deadline=None,
                 deadline_datetime=None, item_id=None, parent_item=None,
                 time_est=None, today=None, value=None):
        # TODO: Add parameter description (!)
        # TODO: Check default values (!)
        # TODO: Implement item predecessors/dependencies (+)
      
        # Compulsory parameters
        self.description = description
        
        # Optional parameters
        self.completed = completed
        self.deadline = deadline
        self.deadline_datetime = deadline_datetime
        self.parent_item = parent_item
        self.time_est = time_est
        self.today = today  # If task --> {True, False}; Else: None
        self.value = value
        
        self.item_id = item_id
        if self.item_id is None:
            self.item_id = self.description

        # Initialize set of goals
        self.goals = set()

        # Initialize expected loss
        self.expected_reward = None

        # Initialize dictionary of maximum future Q-values
        self.future_q = {
            None: None
        }

        # Initialize index
        self.idx = None

        # Initialize list of sub-items
        self.items = deque()
        
        # Initialize next item to execute
        self.next_item = None

        # Initialize number of sub-items
        self.num_items = 0

        # Initialize optimal reward
        self.optimal_reward = np.NINF
        
        # Initialize 0-th state
        self.start_state = list()
        
        # Initialize queue of tasks
        self.tasks = None

        # Initialize today items set
        self.today_items = set()
        
        # Initialize start time
        self.start_time = None
        
        # Initialize time transitions
        self.time_transitions = {
            self.time_est: None
        }
        
        # Initialize computations
        self.already_computed_pruning = 0
        self.small_reward_pruning = 0
        self.total_computations = 0

        # TODO: Some of these might be unnecessary
        # Initialize policy, Q-value function and pseudo-rewards
        self.P = dict()  # {state: {time: action}}
        self.Q = dict()  # {state: {time: {action: {time': value}}}}
        self.PR = dict()  # Pseudo-rewards {(s, t, a): PR(s, t, a)}
        self.tPR = dict()  # Transformed PRs {(s, t, a): tPR(s, t, a)}

        self.F = dict()  # TODO: Describe structure
        self.R = dict()  # TODO: Describe structure

        # Initialize dictionary of Q-values at (s, t): (0, 0)
        self.Q_s0 = dict()

    def __hash__(self):
        return id(self)

    def __str__(self):
        # TODO: Add more items to print (!)
        return f"{self.description} " \
               f"~~{self.time_est} " \
               f"=={self.value} " \
               f"DUE:{self.deadline}"
    
    def add_goal(self, goal):
        self.goals.add(goal)
        
    def add_items(self, items):
        
        # Update number of tasks
        self.num_items += + len(items)

        # Initialize start state
        self.start_state = list(0 for _ in range(self.num_items))
    
        for idx, item in enumerate(items):
    
            # If item has no deadline, set super-item's deadline
            item.set_deadline(self.deadline, compare=True)

            # Set item index | TODO: Check whether there is a conflict in solve... (!)
            item.set_idx(idx)

            if item.is_today() and not item.is_completed():
                self.add_today_item(item)
        
            if item.is_completed():
                self.start_state[idx] = 1
        
            # Add items
            self.items.append(item)
    
        # Convert start state from list to tuple
        self.start_state = tuple(self.start_state)
    
        # Compute goal-level bins | TODO: Increase to > 1 (!)
        self.compute_binning(num_bins=1)
    
        # Sort list of tasks by deadline (break ties with time estimate, idx)
        self.items = list(self.items)
        self.items.sort(
            key=lambda item: (
                item.get_deadline(),
                item.get_time_est(),
                item.get_idx()
            )
        )
        
    def add_today_item(self, item):
        self.today_items.add(item)
        
    def append_task(self, task, left=False):
        if left:
            self.tasks.appendleft(task)
        else:
            self.tasks.append(task)
    
    def compute_binning(self, num_bins=None):
        binned_dist = get_binned_dist(mu=self.time_est, num_bins=num_bins)
    
        bin_means = binned_dist["bin_means"]
        bin_probs = binned_dist["bin_probs"]
    
        self.time_transitions = dict()
    
        for i in range(len(bin_means)):
            mean = int(np.ceil(bin_means[i]))
            prob = bin_probs[i]
        
            self.time_transitions[mean] = prob
            
    def convert_task_list(self):
        self.tasks = list(self.tasks)

    def get_best_action(self, slack_reward, t):
    
        # Initialize best action and Q-value to be the slack action
        best_a = None
        best_q = slack_reward
    
        # Find best action and Q-value in current (state, time)
        for a, q in self.Q_s0.items():
            if best_q <= q:
                best_a = a
                best_q = q
    
        return best_a

    def get_copy(self):
        return deepcopy(self)

    def get_deadline(self):
        return self.deadline

    def get_deadline_datetime(self):
        return self.deadline_datetime

    def get_description(self):
        return self.description

    def get_expected_loss(self):
        return self.expected_reward

    def get_future_q(self, t=None):
        if t is not None:
            return self.future_q[t]
        return self.future_q

    def get_goals(self):
        return self.goals

    def get_id(self):
        return self.item_id

    def get_idx(self):
        return self.idx
    
    def get_items(self):
        return self.items
    
    def get_next_item(self):
        return self.next_item

    def get_num_items(self):
        return self.num_items

    def get_optimal_reward(self):
        return self.optimal_reward
    
    def get_parent_item(self):
        return self.parent_item

    def get_reward(self, beta=0., discount=1.):
        return discount * self.value / (1 + beta)

    def get_start_state(self):
        return self.start_state
    
    def get_task_list(self):
        return self.tasks

    def get_time_est(self):
        return self.time_est
    
    def get_time_transitions(self, t=None):
        if t is None:
            return self.time_transitions
        return self.time_transitions[t]
    
    def get_total_loss(self, loss_rate, cum_discount=None):
        # TODO: Find a better implementation.
        if cum_discount is None:
            return loss_rate * self.time_est
        return loss_rate * cum_discount

    def get_value(self):
        return self.value
    
    def init_task_list(self):
        self.tasks = deque()
    
    def is_completed(self):
        return self.completed
    
    def is_deadline_missed(self, t):
        return t > self.deadline

    def is_today(self):
        return self.today
    
    def print_recursively(self, level=0, indent=2):
        print(f"{' ' * (level * indent)}- {self}")
        for item in self.items:
            item.print_recursively(level + 1, indent=indent)

    def set_completed(self):
        self.completed = True

    def set_deadline(self, deadline, compare=False):
        if self.deadline is None or not compare:
            self.deadline = deadline
        else:
            self.deadline = min(self.deadline, deadline)

    def set_description(self, description):
        self.description = description
        
    def set_expected_reward(self, expected_reward):
    
        # # TODO: Remove (!)
        # Sanity check
        # if self.expected_reward is not None:
        #
        #     # assert self.expected_loss == expected_loss
        #
        #     if self.expected_reward != expected_reward:
        #         print(self.description)
        #         print(self.expected_reward, expected_reward)
        #         print()
        
        self.expected_reward = expected_reward
    
    def set_future_q(self, t, value, compare=False):
        # TODO: Comment this function
        
        self.future_q.setdefault(t, value)
        
        if compare:
            self.future_q[t] = max(self.future_q[t], value)

    def set_idx(self, idx):
        self.idx = idx
        
    def set_next_item(self, next_item):
        self.next_item = next_item
        
    def set_optimal_reward(self, optimal_reward):
        self.optimal_reward = optimal_reward
        
    def set_parent_item(self, parent_item):
        self.parent_item = parent_item
        
    def set_reward(self, reward):
        # TODO: Implement this (!)
        raise NotImplementedError()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_time_est(self, time_est):
        self.time_est = time_est
        
    def set_today(self, today):
        self.today = today
        
    def set_value(self, value):
        self.value = value
        
    def sort_task_list(self):
        self.tasks.sort(
            key=lambda item: (
                item.get_deadline(),
                item.get_time_est(),
                item.get_idx()
            )
        )
        
    # ==================== SOLVERS ====================
    def solve(self, params, available_time=np.PINF,
              start_time=None, verbose=False):
        
        # Initialize start state & time
        s = self.start_state
        t = self.start_state if start_time is None else start_time

        # Initialize Q-function entries
        self.Q.setdefault(s, dict())
        self.Q[s].setdefault(t, dict())

        # Initialize list of tasks to iterate
        tasks_to_iterate = set()
        
        # If the current time is the same with the initial time of the procedure
        if t == self.start_time:

            # If there is no time limit or there is more time than necessary
            if available_time == np.PINF or available_time > self.time_est:

                # Add all tasks
                tasks_to_iterate.update(self.tasks)

            else:

                # Add today tasks
                tasks_to_iterate.update(self.today_items)
                
                # Add other tasks that might be scheduled today
                
                available_time = deepcopy(available_time)

                for task in self.tasks:
                    
                    # If the task is not already marked to be executed
                    if not task.is_completed():
                        
                        # Get task time estimate
                        time_est = task.get_time_est()

                        # Add task to the list of tasks to be executed
                        tasks_to_iterate.add(task)

                        # Decrease available time by the task time estimate
                        available_time -= time_est

                        # Stop the procedure if there is no available time
                        if available_time < 0:
                            break

                # # Append least-optimal action in order to get lower bound on the
                # # reward-shaping value
                # last_task = self.tasks[-1]
                #
                # # Add task item to the list of tasks to be executed
                # tasks_to_iterate.add(last_task)
                
        # If the start time is in the future, find the next uncompleted task
        else:
            
            # Initialize task index
            idx = 0

            # While uncompleted task has been found or list iterated
            while True:
                
                # Get task
                task = self.items[idx]

                # If the task is completed, move to the next one
                if task.is_completed():
                    idx += 1

                # If the task in uncompleted, add it for future execution
                else:
                    tasks_to_iterate.add(self.items[idx])
                    break

        # Start timer
        tic = time.time()
        
        # TODO: Remove (!)
        betas = dict()
        exp_losses = dict()
        durations = dict()
        qs = dict()
        
        # TODO: Remove (!)
        # if t == 0:
        #     for task in tasks_to_iterate:
        #         print(task.get_description())

        for task in tasks_to_iterate:
            
            # Initialize current item
            current_item = task
            
            # Initialize parent item
            parent_item = task.get_parent_item()

            # Until goal node has been reached
            while parent_item is not None:
                
                # Set first item to be executed in the sub-tree
                parent_item.set_next_item(current_item)
                
                # Assign parent item as current item
                current_item = parent_item
            
                # Move up in the item hierarchy
                parent_item = parent_item.get_parent_item()

            # Solve goal
            result = current_item.solve_goal(
                params=params, start_time=t, verbose=verbose
            )
            
            beta = result["beta"]
            exp_loss = result["exp_loss"]
            term_time = result["term_time"]
            duration = int(np.ceil(term_time - t))
            
            # Get discount
            gamma = ToDoList.get_discount(duration)
            
            # Compute Q-value
            q = self.get_reward(beta=beta, discount=gamma) + exp_loss
            
            # # TODO: Remove (!)
            # print(self.description)
            # print("Starting time:", t)
            # print("Beta:", beta)
            # print("Expected loss:", exp_loss)
            # print("Termination time:", term_time)
            # print("Expected duration:", term_time - t)
            # print("Q-value:", q)
            # print()
            
            # desc = self.get_description()
            #
            # if desc in betas.keys():
            #     # print("Beta")
            #     assert np.isclose(betas[desc], beta, atol=1e-9)
            # else:
            #     betas[desc] = beta
            #
            # if desc in durations.keys():
            #     # print("Duration")
            #     assert np.isclose(durations[desc], duration, atol=1e-9)
            # else:
            #     durations[desc] = duration
            #
            # if desc in exp_losses.keys():
            #     # print("Expected loss")
            #     assert np.isclose(exp_losses[desc], exp_loss, atol=1e-9)
            # else:
            #     exp_losses[desc] = exp_loss
            #
            # if desc in qs.keys():
            #     # print("Q-value")
            #     assert np.isclose(qs[desc], q, atol=1e-9)
            # else:
            #     qs[desc] = q

            # Store initial-state Q-value (if initial time applies)
            if t == self.start_time:
                self.Q_s0[task] = q
            
            # Initialize next item
            next_item = current_item.get_next_item()
            
            # Get item index
            a = self.next_item.get_idx()
            
            # Assign default value
            self.Q[s][t].setdefault(a, np.NINF)
            
            # Store Q-value | TODO: Potentially unnecessary (?!)
            self.Q[s][t][a] = max(self.Q[s][t][a], q)
            
            # Until leaf node has been reached
            while next_item is not None:
                
                # Set next item to be None
                current_item.set_next_item(None)
                
                # Assign next item to the current item
                current_item = next_item
                
                # Get next item
                next_item = current_item.get_next_item()

        if verbose and t == self.start_time:
            print(
                f"{t:>3d} | "
                f"Number of tasks: {len(tasks_to_iterate)} | "
                f"Time: {time.time() - tic:.2f}\n"
            )

        # Return maximum Q-value
        return max(self.Q[s][t].values())

    def solve_goal(self, params, beta=None, start_time=None, verbose=False):
    
        # Initialize start state & time
        s = self.start_state
        t = self.start_state if start_time is None else start_time

        # Initialize state
        curr_state = {
            "s":    deepcopy(s),
            "t":    t,
            "idx":  0,
            "beta": 0. if beta is None else beta  # TODO: Correct (?!)
        }
    
        # Get next item to be executed
        next_item = self.get_next_item()
        
        # Return dictionary of information
        return self.solve_branching(curr_state=curr_state, next_item=next_item,
                                    params=params, verbose=verbose)
        
    def solve_branching(self, curr_state, params, next_item=None, verbose=False):
        """
            - s: current state
            - t: current time
            - a: current action | taken at (s, t)
            - s_: next state (consequence of taking a in (s, t))
            - t_: next time (consequence of taking a in (s, t) with dur. x)
            - a_: next action | taken at (s_, t_)
            - r: reward of r(s, a, s') with length (t_ - t)
        """
        s = curr_state["s"]
        t = curr_state["t"]
        idx = curr_state["idx"]

        # Initialize next action index
        a = None
        
        if next_item is None:
        
            # Get next uncompleted item (if one exists)
            while idx < self.num_items:
            
                # Get next item from the queue
                item = self.items[idx]
            
                # Get next action index
                a = item.get_idx()
            
                # If the next item in the queue is completed
                if s[a]:
                    idx += 1
                    
                else:
                    next_item = item
                    break
    
        else:
            
            # Set action to be the index of the next item
            a = next_item.get_idx()

        if verbose:
            print(
                f"\nCurrent item-level state: {s} | "
                f"Current time: {t:>3d} | "
                f"Task index: {a if a is not None else '-'} | "
                f"Idx: {idx} | "
                ,
                end=""
            )
            
        # Initialize result variables
        beta = 0
        exp_loss = 0
        term_time = 0

        if next_item is not None:
    
            # Get next state by taking action in state
            s_ = ToDoList.exec_action(s, a)

            # Generate next item-level state (by updating current state)
            next_state = deepcopy(curr_state)
            next_state["s"] = s_
            next_state["idx"] = idx

            # If next_item is a task node
            if next_item.get_num_items() == 0:
    
                # Get deadline time for next item
                task_deadline = next_item.get_deadline()

                # Get time transitions of the next state
                time_transitions = next_item.get_time_transitions().items()
                
                # Initialize expected loss
                exp_task_loss = 0
                
                for time_est, prob_t_ in time_transitions:
    
                    # Increase total-computations counter
                    self.total_computations += 1
    
                    # Make time transition
                    t_ = t + time_est
                    next_state["t"] = t_

                    # Get cumulative discount w.r.t. item duration
                    cum_discount = ToDoList.get_cum_discount(time_est)
        
                    # Calculate total loss for next action (immediate "reward")
                    r = next_item.get_total_loss(loss_rate=params["loss_rate"],
                                                 cum_discount=cum_discount)

                    # Add deadline to the missed deadlines if not attained
                    if next_item.is_deadline_missed(t_):
                        
                        # Compute total penalty for missing item deadline
                        total_penalty = params["penalty_rate"] * \
                                        (t_ - task_deadline)
        
                        # Update penalty for the next state
                        next_state["beta"] += total_penalty
                        
                        beta += total_penalty
                        
                    # Solve branching of the next state
                    result = self.solve_branching(next_state, params,
                                                  verbose=verbose)
                    
                    # Update expected loss for (state, time, action)
                    exp_task_loss += prob_t_ * r
    
                    """ Compute total reward for the current state-time action
                        Immediate + Expected future reward """
                    
                    # If next best action is a terminal state
                    # Single time-step discount (MDP)
                    # if a_ is None:
                    #     total_reward = q + q_
                    #
                    # Otherwise
                    # else:
                    #     Single time-step discount (MDP)
                    #     total_reward = q + ToDoList.get_discount(1) * q_
            
                    # Compute discount factor
                    gamma = ToDoList.get_discount(time_est)
            
                    # Update expected future penalty
                    beta += prob_t_ * result["beta"]
                    
                    # Update expected future loss
                    exp_loss += prob_t_ * (r + gamma * result["exp_loss"])
                    
                    # Update expected termination time
                    term_time += prob_t_ * result["term_time"]
                    
                # Assign expected loss
                next_item.set_expected_reward(exp_task_loss)

            # If next_item is a sub-goal node
            else:
                
                # Get time estimate of the sub-goal
                time_est = next_item.get_time_est()
                
                # Make time transition
                t_ = t + time_est
                
                result = next_item.solve_goal(
                    params=params, beta=deepcopy(curr_state["beta"]),
                    start_time=t_, verbose=verbose
                )
                
                # TODO: Allow other probability values
                prob = 1

                # Update expected future penalty
                beta += prob * result["beta"]

                # Update expected future loss
                exp_loss += prob * result["exp_loss"]

                # Generate next item-level state (by updating current state)
                next_state = deepcopy(curr_state)
                next_state["s"] = s_
                next_state["t"] = t_
                next_state["idx"] = idx

                # Solve branching of the next state
                result = self.solve_branching(next_state, params,
                                              verbose=verbose)
                
                # Compute discount factor
                gamma = ToDoList.get_discount(time_est)

                # TODO: Allow other probability values
                prob = 1

                # Update expected future penalty
                beta += prob * result["beta"]
        
                # Update expected future loss
                exp_loss += prob * gamma * result["exp_loss"]
        
                # Update expected termination time
                term_time = prob * result["term_time"]

        # Terminal state
        else:
            
            # Initialize termination time
            term_time = t

        # Return Q-value of the current state
        return {
            "beta":      beta,
            "exp_loss":  exp_loss,
            "term_time": term_time
        }
    
    
class ToDoList:
    
    # Class attributes
    DISCOUNTS = None
    CUM_DISCOUNTS = None
    
    def __init__(self, goals, end_time=np.PINF, gamma=1.0, loss_rate=0.,
                 num_bins=1, penalty_rate=0., planning_fallacy_const=1.,
                 slack_reward_rate=0, start_time=0):
        """

        Args:
            goals: [Goal]
            end_time: End time of the MDP (i.e. horizon)
            gamma: Discount factor
            slack_reward_rate: Unit-time reward for slack-off action.
            start_time:  Starting time of the MDP
        """
        
        self.description = "__TO_DO_LIST__"
        self.end_time = end_time
        self.gamma = gamma
        self.loss_rate = loss_rate
        self.num_bins = num_bins
        self.penalty_rate = penalty_rate
        self.planning_fallacy_const = planning_fallacy_const
        self.slack_reward_rate = slack_reward_rate
        self.start_time = start_time
        
        # Initialize list of goals
        self.goals = deque()
        
        # Set number of goals
        self.num_goals = 0

        # Initialize 0-th state
        self.start_state = tuple()

        # Calculate total time estimate of the to-do list
        self.total_time_est = 0
        
        # Add goals to the to-do list
        self.add_goals(goals)
        
        # "Cut" horizon in order to reduce the number of computations
        self.end_time = min(self.end_time, self.total_time_est)
        
        # Generate discounts | TODO: Add epsilon as input parameter (!)
        ToDoList.generate_discounts(epsilon=0., gamma=self.gamma,
                                    horizon=self.end_time, verbose=False)
        
        # Slack-off reward
        self.slack_reward = self.compute_slack_reward()

        # Initialize policy, Q-value function and pseudo-rewards
        self.P = dict()  # Optimal policy {state: action}
        self.Q = dict()  # Action-value function {state: {action: value}}
        self.PR = dict()  # Pseudo-rewards {(s, t, a): PR(s, t, a)}
        self.tPR = dict()  # Transformed PRs {(s, t, a): tPR(s, t, a)}
        
        self.F = dict()
        self.R = dict()

        # Initialize computation counters
        self.small_reward_pruning = 0
        self.already_computed_pruning = 0
        self.total_computations = 0
        
    @classmethod
    def generate_discounts(cls, epsilon=0., gamma=1., horizon=1, verbose=False):
        tic = time.time()

        ToDoList.DISCOUNTS = deque([1.])
        ToDoList.CUM_DISCOUNTS = deque([0., 1.])
        
        for t in range(1, horizon + 1):
            last_power_value = ToDoList.DISCOUNTS[-1] * gamma
            if last_power_value > epsilon:
                ToDoList.DISCOUNTS.append(last_power_value)
                ToDoList.CUM_DISCOUNTS.append(
                    ToDoList.CUM_DISCOUNTS[-1] + last_power_value)
            else:
                break
    
        ToDoList.DISCOUNTS = list(ToDoList.DISCOUNTS)
        ToDoList.CUM_DISCOUNTS = list(ToDoList.CUM_DISCOUNTS)
        
        toc = time.time()
        
        if verbose:
            print(f"\nGenerating powers took {toc - tic:.2f} seconds!\n")
            
        return ToDoList.DISCOUNTS, ToDoList.CUM_DISCOUNTS

    @classmethod
    def get_cum_discount(cls, t):
        n = len(ToDoList.CUM_DISCOUNTS)
        cum_discount = ToDoList.CUM_DISCOUNTS[min(t, n - 1)]
    
        if t < 0:
            raise Exception("Negative time value for cumulative discount is "
                            "not allowed!")
        return cum_discount

    @classmethod
    def get_discount(cls, t):
        n = len(ToDoList.DISCOUNTS)
        discount = ToDoList.DISCOUNTS[min(t, n - 1)]
        
        if t < 0:
            raise Exception("Negative time value for discount is not allowed!")
        
        return discount

    @staticmethod
    def get_policy(obj, s=None, t=None):
        if s is not None:
            if t is not None:
                return obj.P[s][t]
            return obj.P[s]
        return obj.P

    @staticmethod
    def get_pseudo_rewards(obj, s=None, t=None, a=None, transformed=False):
        """
        Pseudo-rewards are stored as a 3-level dictionaries:
            {state: {time: {action: pseudo-reward}}}
        """
        if transformed:
            if s is not None:
                if t is not None:
                    if a is not None:
                        return obj.tPR[s][t][a]
                    return obj.tPR[s][t]
                return obj.tPR[s]
            return obj.tPR
    
        if s is not None:
            if t is not None:
                if a is not None:
                    return obj.PR[s][t][a]
                return obj.PR[s][t]
            return obj.PR[s]
        return obj.PR

    @staticmethod
    def get_q_values(obj, s=None, t=None, a=None, t_=None):
        if s is not None:
            if t is not None:
                if a is not None:
                    if t_ is not None:
                        return obj.Q[s][t][a][t_]
                    return obj.Q[s][t][a]
                return obj.Q[s][t]
            return obj.Q[s]
        return obj.Q

    @classmethod
    def exec_action(cls, s, a):
        s_ = list(s)
        s_[a] = 1
        return tuple(s_)

    @classmethod
    def max_from_dict(cls, d: dict):
        max_a = None
        max_q = np.NINF
        
        for a in d.keys():
            
            # Get expected Q-value
            q = d[a]
            
            # If a is a better action than max_a
            if max_q < q:
                max_a = a
                max_q = q
            
            # Prefer slack-off action over working on a goal
            elif max_q == q and a == -1:
                max_a = a
                max_q = q
    
        return max_a, max_q

    def add_goals(self, goals):
        
        # Update number of goals
        self.num_goals += len(goals)

        # Initialize start state
        self.start_state = list(0 for _ in range(self.num_goals))

        for idx, goal in enumerate(goals):
        
            # Increase total time estimate of the to-do list
            self.total_time_est += goal.get_time_est()
        
            # Set goal index
            goal.set_idx(idx)
            
            # Set start time
            goal.set_start_time(self.start_time)
            
            # Add goal to the list of goals
            self.goals.append(goal)
            
            if goal.is_completed():
                self.start_state[idx] = 1

        # Convert list to tuple
        self.start_state = tuple(self.start_state)

    def compute_slack_reward(self):
        if self.slack_reward_rate == 0:
            return 0
    
        if self.gamma < 1:
            return self.slack_reward_rate * (1 / (1 - self.gamma))
    
        return np.PINF

    def get_description(self):
        return self.description

    def get_end_time(self):
        return self.end_time

    def get_gamma(self):
        return self.gamma
    
    def get_goals(self):
        return self.goals
    
    def get_loss_rate(self):
        return self.loss_rate

    def get_num_bins(self):
        return self.num_bins

    def get_num_goals(self):
        return self.num_goals

    def get_optimal_policy(self, state=None):
        """
        Returns the mapping of state to the optimal policy at that state
        """
        if state is not None:
            return self.P[state]
        return self.P

    def get_penalty_rate(self):
        return self.penalty_rate

    def get_planning_fallacy_const(self):
        return self.planning_fallacy_const

    def get_start_state(self):
        return self.start_state

    def get_start_time(self):
        return self.start_time
    
    def get_slack_reward(self):
        return self.slack_reward

    def get_slack_reward_rate(self):
        return self.slack_reward_rate

    def set_gamma(self, gamma):
        assert 0 < gamma <= 1
        self.gamma = gamma

    def set_loss_rate(self, loss_rate):
        self.loss_rate = loss_rate

    def set_num_bins(self, num_bins):
        self.num_bins = num_bins

    def set_penalty_rate(self, penalty_rate):
        self.penalty_rate = penalty_rate

    def set_planning_fallacy_const(self, planning_fallacy_const):
        self.planning_fallacy_const = planning_fallacy_const

    def solve(self, available_time=np.PINF, verbose=False):
        
        params = {
            "loss_rate": self.loss_rate,
            "penalty_rate": self.penalty_rate
        }
    
        def solve_next_goals(curr_state, verbose=False):
            """
                - s: current state
                - t: current time
                - a: current action | taken at (s, t)
                - s_: next state (consequence of taking a in (s, t))
                - t_: next time (consequence of taking a in (s, t) with dur. x)
                - a_: next action | taken at (s_, t_)
                - r: reward of r(s, a, s') with length (t_ - t)
            """
            s = curr_state["s"]
            t = curr_state["t"]
            
            if verbose:
                print(
                    f"Current state: {s} | "
                    f"Current time: {t:>3d} | "
                    , end=""
                )
                
            # Initialize next goal
            next_goal = None

            # Initialize policy entries for state and (state, time))
            self.P.setdefault(s, dict())
            self.P[s].setdefault(t, dict())

            # Initialize Q-function entry for the (state, time) pair
            self.Q.setdefault(s, dict())
            self.Q[s].setdefault(t, dict())

            # Initialize reward entries for state and (state, time))
            self.R.setdefault(s, dict())
            self.R[s].setdefault(t, dict())
            
            # Find the next uncompleted goal
            for goal_idx in range(self.num_goals):
    
                # If the goal with index goal_idx is not completed
                if s[goal_idx] == 0:
    
                    # Increase total-computations counter
                    self.total_computations += 1
    
                    # Set action to be the corresponding goal index
                    a = goal_idx

                    # Generate next state
                    s_ = ToDoList.exec_action(s, a)

                    # Get next Goal object
                    next_goal = self.goals[goal_idx]
                    
                    # Get goal's time estimate
                    time_est = next_goal.get_time_est()

                    # Move for "expected goal time estimate" units in the future
                    t_ = t + time_est

                    # TODO: Probability of transition to next state
                    prob = 1

                    # Calculate total reward for next action
                    # result = next_goal.solve(start_time=t, verbose=verbose)
                    # r = result["r"]
                    
                    # Initialize Q-values for state' and time'
                    self.Q.setdefault(s_, dict())
                    self.Q[s_].setdefault(t_, dict())

                    # The computation has already been done --> Prune!
                    if a in self.Q[s][t].keys():
        
                        # Increase already-computed-pruning counter
                        self.already_computed_pruning += 1

                        if verbose:
                            print(f"Transition (s, t, a, t') {(s, t, a, t_)} "
                                  f"already computed.")
                            
                    # Explore the next goal-level state
                    else:
    
                        # Initialize expected value for action
                        self.Q[s][t].setdefault(a, 0)
    
                        # Initialize entry for (state, time, action)
                        self.R[s][t].setdefault(a, 0)

                        r = next_goal.solve(params=params,
                                            available_time=available_time,
                                            start_time=t, verbose=verbose)
                        
                        self.R[s][t][a] += prob * r
    
                        if verbose:
                            print()
                        
                        # Generate next goal-level state
                        state_dict = {
                            "s": s_,
                            "t": t_
                        }
                        
                        # Explore the next goal-level state
                        solve_next_goals(state_dict, verbose=verbose)
                        
                        # Get best action and its respective for (state', time')
                        a_, r_ = ToDoList.max_from_dict(self.Q[s_][t_])
    
                        # Store policy for the next (state, time) pair
                        self.P[s_][t_] = a_
                        
                        # Store future Q-value
                        next_goal.set_future_q(t_, r_, compare=True)
                        
                        # Compute total reward for the current state-time action as
                        # immediate + (discounted) expected future reward
                        # - Single time-step discount (MDP)
                        # total_reward = r + self.gamma ** next_goal.num_tasks * r_
                        
                        # - Multiple time-step discount (SMDP)
                        total_reward = r + self.get_discount(time_est) * r_
                        
                        # TODO: Probability of transition to next state
                        prob = 1
    
                        # Add more values to the expected value
                        self.Q[s][t][a] += prob * total_reward
                        
            # ===== Terminal state =====
            if next_goal is None:
                
                # Initialize dictionary for the terminal state Q-value
                self.Q[s][t].setdefault(None, 0)

                # Compute reward for reaching terminal state s in time t
                self.R[s][t].setdefault(None, 0)

        # Iterate procedure
        s = tuple(0 for _ in range(self.num_goals))
        t = self.start_time

        curr_state = {
            "s": s,
            "t": t
        }

        # Start iterating
        solve_next_goals(curr_state, verbose=verbose)
        
        # Get best action in the start state and its corresponding reward
        a, r = ToDoList.max_from_dict(self.Q[s][t])
        
        # Store policy for the next (state, time) pair
        self.P[s][t] = a

        return {
            "P": self.P,
            "Q": self.Q,
            "s": s,
            "t": t,
            "a": a,
            "r": r
        }
