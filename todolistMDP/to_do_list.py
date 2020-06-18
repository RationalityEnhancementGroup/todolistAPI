import itertools
import numpy as np
import random
import time

from abc import abstractmethod
from collections import defaultdict, deque
from copy import deepcopy
from math import ceil, gcd
from todolistMDP.zero_trunc_poisson import get_binned_distrib
from pprint import pprint
from scipy.stats import poisson
from todolistMDP import mdp


class Item:
    def __init__(self, description, hard_deadline=True, idx=None, item_id=None,
                 loss_rate=None, num_bins=None, planning_fallacy_const=None,
                 rewards=None, time_est=None, time_precision=None,
                 time_support=None, unit_penalty=None):
        self.description = description
        self.hard_deadline = hard_deadline
        self.idx = idx
        self.unit_penalty = unit_penalty
        
        self.item_id = item_id
        if self.item_id is None:
            self.item_id = self.description
            
        self.loss_rate = loss_rate
        self.num_bins = num_bins
        self.optimal_reward = np.NINF
        self.planning_fallacy_const = planning_fallacy_const
        self.rewards = rewards
        self.time_est = time_est
        
        self.latest_deadline_time = max(rewards.keys())

        self.lowest_time_est = self.time_est
        self.highest_time_est = self.time_est
        self.time_precision = time_precision
        self.time_support = time_support
        self.total_time_support = 1  # TODO: Is this good init?

        self.time_transition_prob = {
            self.time_est: None
        }

    def __hash__(self):
        return id(self)

    def __str__(self):
        return f"Description: {self.description}\n" \
               f"Index: {self.idx}\n" \
               f"Latest deadline time: {self.latest_deadline_time}\n"  \
               f"Optimal reward: {self.optimal_reward}\n" \
               f"Time estimate: {self.time_est}\n"

    def compute_binning(self):
        # TODO: Comment function...
        
        binned_distrib = get_binned_distrib(mu=self.time_est,
                                            num_bins=self.num_bins)
    
        bin_means = binned_distrib["bin_means"]
        bin_probs = binned_distrib["bin_probs"]
    
        self.time_transition_prob = dict()
    
        for i in range(len(bin_means)):
            mean = int(np.ceil(bin_means[i]))
            prob = bin_probs[i]
        
            self.time_transition_prob[mean] = prob

    def get_copy(self):
        return deepcopy(self)

    def get_deadline(self, t=0):
        if self.latest_deadline_time is None or t > self.latest_deadline_time:
            return None
        
        times = sorted(self.rewards.keys())
        return next(val for x, val in enumerate(times) if val >= t)

    def get_description(self):
        return self.description

    def get_id(self):
        return self.item_id

    def get_idx(self):
        return self.idx
    
    def get_latest_deadline_time(self):
        return self.latest_deadline_time
    
    def get_loss_rate(self):
        return self.loss_rate

    def get_num_bins(self):
        return self.num_bins

    def get_optimal_reward(self):
        return self.optimal_reward
    
    def get_planning_fallacy_const(self):
        return self.planning_fallacy_const

    def get_reward(self, beta=0., discount=1., t=0):
        # If the latest deadline has not been met, return no reward
        if t > self.latest_deadline_time:
            if self.hard_deadline:
                return 0
            else:
                # Add penalty for not attaining goal's deadline
                beta += self.unit_penalty
                
                # Get discounted reward
                reward = discount * self.rewards[self.latest_deadline_time]
                
                return reward / (1 + beta * (t - self.latest_deadline_time))

        # Otherwise, get the reward for the next deadline that has been met
        deadline = self.get_deadline(t=t)
        return self.rewards[deadline] * discount / (1 + beta)

    def get_time_est(self):
        return self.time_est
    
    def get_time_transition_prob(self, t=None):
        if t is None:
            return self.time_transition_prob
        return self.time_transition_prob[t]
    
    def get_total_loss(self, discount=None):
        # TODO: Find a better implementation.
        if discount is None:
            return self.loss_rate * self.time_est
        return self.loss_rate * discount
    
    def get_unit_penalty(self):
        return self.unit_penalty
    
    def is_hard_deadline(self):
        return self.hard_deadline
    
    def set_idx(self, idx):
        self.idx = idx
        
    def set_loss_rate(self, loss_rate):
        self.loss_rate = loss_rate
        
    def set_num_bins(self, num_bins):
        self.num_bins = num_bins

    def set_optimal_reward(self, optimal_reward):
        self.optimal_reward = optimal_reward
        
    def set_planning_fallacy_const(self, planning_fallacy_const):
        self.planning_fallacy_const = planning_fallacy_const

    def set_time_est(self, time_est):
        self.time_est = time_est
        
        # Compute time support
        self.compute_binning()
        
    def set_time_precision(self, time_precision):
        self.time_precision = time_precision
        
    # def set_time_support(self, time_support):
    #     self.time_support = time_support
    #
    #     # Compute standard deviation of time estimates
    #     std = np.sqrt(self.time_est)
    #     lb_std = max(1, int(self.time_est - 3 * std))
    #     ub_std = int(self.time_est + 3 * std)
    #
    #     self.highest_time_est = self.time_est
    #     self.lowest_time_est = self.time_est
    #
    #     if self.time_support is not None:
    #
    #         # Store the probability of task having a duration of 0 minutes
    #         # Time estimate cannot be 0!
    #         zero_prob = poisson.pmf(0, mu=self.time_est)
    #
    #         # self.total_time_support = poisson.pmf(self.time_est, mu=self.time_est)
    #
    #         self.time_support = min(self.time_support, 1 - zero_prob)
    #
    #         update = True
    #
    #         # Set lower and upper bounds
    #         # while self.total_time_support < self.time_support:
    #         while update:
    #
    #             # Reset update
    #             update = False
    #
    #             # Store total probability from the previous iteration
    #             # prev_prob = self.total_time_support
    #
    #             # If the lower bound of time estimates is allowed
    #             if self.lowest_time_est - self.time_precision >= lb_std:
    #                 self.lowest_time_est -= self.time_precision
    #                 self.time_transition_prob[self.lowest_time_est] = None
    #
    #                 update = True
    #
    #             # If the upper bound of time estimates is allowed
    #             if self.highest_time_est + self.time_precision <= ub_std:
    #                 self.highest_time_est += self.time_precision
    #                 self.time_transition_prob[self.highest_time_est] = None
    #
    #                 update = True
    #
    #             # self.total_time_support = \
    #             #     poisson.cdf(self.highest_time_est, self.time_est) - \
    #             #     zero_prob
    #             #     # poisson.cdf(self.lowest_time_est-1, self.time_est)
    #
    #             # If there is very small contribution to the total probability
    #             # if abs(self.total_time_support - prev_prob) < 1e-6:
    #             #     break
    #
    #     # Initialize a normalizing constant
    #     normalizer = 0
    #
    #     # Initialize lower bound of the bin
    #     lb = 0
    #
    #     # For each next upper bound
    #     for ub in sorted(list(self.time_transition_prob.keys())):
    #
    #         # Get probability of the current bin
    #         prob = poisson.cdf(ub, self.time_est) - poisson.cdf(lb, self.time_est)
    #
    #         # Assign probability to the current bin
    #         self.time_transition_prob[ub] = prob
    #
    #         # Add probability to the normalizing constant
    #         normalizer += prob
    #
    #         # Move lower bound of the next bin
    #         lb = ub
    #
    #     # Normalize bin probabilities to have a proper probability distribution
    #     for ub in self.time_transition_prob.keys():
    #         self.time_transition_prob[ub] /= normalizer
    #
    #     # pprint(self.time_transition_prob)
        
    def set_unit_penalty(self, unit_penalty):
        self.unit_penalty = unit_penalty
        

class Task(Item):
    """
    TODO:
        - Task predecessors/dependencies
    """
    
    def __init__(self, description, deadline=None, loss_rate=None, reward=0,
                 task_id=None, time_est=0, prob=1., time_precision=None,
                 time_support=None, unit_penalty=None):
        super().__init__(
            description=description,
            item_id=task_id,
            loss_rate=loss_rate,
            rewards={deadline: reward},
            time_est=time_est,
            time_precision=time_precision,
            time_support=time_support,
            unit_penalty=unit_penalty
        )
        
        self.deadline = deadline
        self.reward = reward
        self.goals = set()
        self.prob = prob
        
    # def __eq__(self, other):
    #     return self.time_est == other.time_est
    #
    # def __ne__(self, other):
    #     return self.time_est != other.time_est
    #
    # def __ge__(self, other):
    #     return self.time_est >= other.time_est
    #
    # def __gt__(self, other):
    #     return self.time_est > other.time_est
    #
    # def __le__(self, other):
    #     return self.time_est <= other.time_est
    #
    # def __lt__(self, other):
    #     return self.time_est < other.time_est

    def __str__(self):
        return super().__str__() + \
            f"Probability: {self.prob}\n"

    def add_goal(self, goal):
        self.goals.add(goal)
        
    def get_highest_time_est(self):
        return self.highest_time_est
        
    def get_lowest_time_est(self):
        return self.lowest_time_est

    def get_prob(self):
        return self.prob
    
    def get_time_precision(self):
        return self.time_precision
    
    def get_time_support(self):
        return self.time_support
    
    def get_total_reward(self):
        return self.get_reward() * self.time_est

    def set_deadline(self, deadline, compare=False):
        if self.deadline is None or not compare:
            self.deadline = deadline
        else:
            self.deadline = min(self.deadline, deadline)
            
        self.update_latest_deadline_time()
        self.update_rewards_dict()
        
    def set_reward(self, reward):
        self.reward = reward
        self.update_rewards_dict()
        
    def update_latest_deadline_time(self):
        self.latest_deadline_time = self.deadline
        
    def update_rewards_dict(self):
        self.rewards = {self.deadline: self.reward}


class Goal(Item):
    
    def __init__(self, description, gamma=None, goal_id=None,
                 hard_deadline=True, loss_rate=0, num_bins=1, rewards=None,
                 planning_fallacy_const=1., slack_reward=None,
                 task_unit_penalty=0., tasks=None, time_precision=1,
                 time_support=None, unit_penalty=0.):
        super().__init__(
            description=description,
            hard_deadline=hard_deadline,
            item_id=goal_id,
            loss_rate=loss_rate,
            num_bins=num_bins,
            planning_fallacy_const=planning_fallacy_const,
            rewards=rewards,
            time_est=0,
            time_precision=time_precision,
            time_support=time_support,
            unit_penalty=unit_penalty
        )
        
        self.slack_reward = slack_reward
        self.task_unit_penalty = task_unit_penalty
        
        # Initialize task list
        self.gamma = gamma
        self.sorted_tasks_by_time_est = None
        self.sorted_tasks_by_deadlines = None
        
        self.tasks = tasks
        if tasks is not None:
            self.add_tasks(tasks)

        self.num_tasks = len(self.tasks)

        # Initialize policy, Q-value function and pseudo-rewards
        self.P = dict()  # {state: {time: action}}
        self.Q = dict()  # {state: {time: {action: {time': value}}}}
        self.PR = dict()  # Pseudo-rewards {(s, t, a): PR(s, t, a)}
        self.tPR = dict()  # Transformed PRs {(s, t, a): tPR(s, t, a)}

        self.F = dict()
        self.R = dict()
        self.R_ = dict()

        # Initialize computations
        self.small_reward_pruning = 0
        self.already_computed_pruning = 0
        self.total_computations = 0

        # Initialize slack-off action
        self.slack_action = Task("__SLACK-OFF__", deadline=np.PINF,
                                 reward=slack_reward, time_est=1)

        # Initialize highest negative reward
        self.highest_negative_reward = np.NINF

    def __str__(self):
        return super().__str__() + \
            f"Rewards: {self.rewards}\n" + \
            f"Slack reward: {self.slack_reward}\n"
            
    def add_tasks(self, tasks):
        for idx, task in enumerate(tasks):
            
            # If task has no deadline, set goal's deadline
            if task.get_deadline() is None:
                task.set_deadline(self.latest_deadline_time)  # TODO: Check (!)
                
            if task.get_loss_rate() is None:
                task.set_loss_rate(self.loss_rate)

            if task.get_num_bins() is None:
                task.set_num_bins(self.num_bins)

            if task.get_planning_fallacy_const() is None:
                task.set_planning_fallacy_const(self.planning_fallacy_const)

            if task.get_time_precision() is None:
                task.set_time_precision(self.time_precision)
                
            # if task.get_time_support() is None:
            #     task.set_time_support(self.time_support)
            
            if task.get_unit_penalty() is None:
                task.set_unit_penalty(self.task_unit_penalty)
                
            # TODO:
            #     - Pass number of bins
            #     - Pass planning fallacy
            
            # Connect task with goal
            task.add_goal(self)
            
            # Adjust task time estimate by the planning-fallacy constant
            task_time_est = ceil(task.get_time_est() *
                                 self.planning_fallacy_const)
            task.set_time_est(task_time_est)
            
            # Add time estimate
            self.time_est += task_time_est
            
            # Set task index
            task.set_idx(idx)
            
        # Sorted list of tasks by time estimate
        self.sorted_tasks_by_time_est = deepcopy(tasks)
        self.sorted_tasks_by_time_est.sort(key=lambda item: item.get_time_est())
        
        # Sorted list of tasks by deadline (break ties with time estimate, idx)
        self.sorted_tasks_by_deadlines = deepcopy(self.sorted_tasks_by_time_est)
        self.sorted_tasks_by_deadlines.sort(
            key=lambda item: (
                item.get_deadline(),
                item.get_time_est(),
                item.get_idx()
            )
        )

    def check_missed_task_deadline(self, deadline, t):
        return t > deadline != self.latest_deadline_time

    # def compute_penalties(self, missed_deadlines, t):
    #     self.deadline_mode = "hard"  # "soft"
    #
    #     # Initialize total penalty
    #     total_penalty = 0
    #
    #     cum_discount_t = ToDoList.get_cum_discount(t)
    #
    #     for deadline in missed_deadlines:
    #
    #         # TODO: Calculate penalty | Sum over all times?!
    #         if t > curr_state["missed_deadlines"]:
    #             penalty = self.penalty
    #         cum_discount_deadline = ToDoList.get_cum_discount(deadline)
    #
    #         if t > deadline:
    #             # penalty = self.penalty * (cum_discount_t - cum_discount_deadline)
    #             penalty = - self.get_reward(self.latest_deadline_time-1) / (1 + t - deadline)
    #             penalty *= (cum_discount_t - cum_discount_deadline)
    #
    #     return total_penalty

    def compute_pseudo_rewards(self, start_time=0, loc=0., scale=1.):
        """
        Computes pseudo-rewards.
        
        TODO: Add transformation here.
        """
        standardizing_reward = self.highest_negative_reward
        if self.highest_negative_reward == np.NINF:
            standardizing_reward = 0
        
        # Get time-shift discount
        discount = ToDoList.get_discount(start_time)

        for s in self.Q.keys():
    
            self.PR.setdefault(s, dict())
            self.F.setdefault(s, dict())

            for t in self.Q[s].keys():
    
                self.PR[s].setdefault(t, dict())
                self.F[s].setdefault(t, dict())
                
                """ Incorporate slack action for (state, time) """
                # Add slack reward to the dictionary
                slack_reward = self.compute_slack_reward(0)
    
                # Add slack reward in the Q-values
                self.Q[s][t].setdefault(-1, dict())
                self.Q[s][t][-1]["E"] = slack_reward
    
                # Add slack reward in the rewards
                self.R[s][t].setdefault(-1, dict())
                self.R[s][t][-1]["E"] = slack_reward
                
                # The best possible (a)ction and (q)-value in state s
                best_a, best_q = ToDoList.max_from_dict(self.Q[s][t])
                best_q *= discount
                
                for a in self.Q[s][t].keys():
                    
                    self.PR[s][t].setdefault(a, dict())
                    self.F[s][t].setdefault(a, dict())
    
                    if a is None or a == -1:

                        # Set pseudo-reward of terminal state to 0 [Ng, 1999]
                        self.PR[s][t][a].setdefault("E", 0)
                        self.F[s][t][a].setdefault("E", 0)
                        self.R[s][t][a].setdefault("E", 0)
                        
                        # TODO: Comment this part
                        s_ = s
                        t_ = np.PINF
                        gamma = 1.
                        
                        if a == -1:
                            curr_task = self.slack_action
                        else:
                            curr_task = None
                        
                        a_ = None
                        q_ = 0
                        
                    else:
                        
                        # Get current Task object
                        curr_task = self.tasks[a]
                        
                        # Move to the next state
                        s_ = ToDoList.exec_action(s, a)
    
                        # Get expected task time estimate
                        time_est = curr_task.get_time_est()
    
                        # Make time transition
                        t_ = t + time_est
    
                        # Get gamma for the next transition
                        gamma = ToDoList.get_discount(time_est)
                        
                        print(s, t, a, s_, t_)
                        pprint(self.Q)
                        print()
    
                        # Get optimal action in the next state
                        a_, q_ = ToDoList.max_from_dict(self.Q[s_][t_])
                        
                    # self.R.setdefault(s_, dict())
                    self.R[s_].setdefault(t_, dict())
                    self.R[s_][t_].setdefault(a_, dict())
    
                    # Expected Q-value of the next state
                    q_ *= discount * gamma
                    # q_ = self.Q[s_][t_][a_]["E"] * discount * gamma
    
                    # Expected Q-value of the current state
                    # q = self.Q[s][t][a]["E"] * discount
                    
                    # Compute value of the reward-shaping function
                    # f = q_ - q
                    f = q_ - best_q
                    
                    # Standardize rewards s.t. negative rewards <= 0
                    # f -= standardizing_reward * discount
                    
                    # Make affine transformation of the reward-shaping function
                    f = scale * f + loc
                    
                    # Get expected reward for (state, time, action)
                    r = self.R[s][t][a]["E"]

                    # Store reward-shaping value
                    self.F[s][t][a]["E"] = f

                    # Calculate pseudo-reward
                    pr = f + r
                    
                    # Set pseudo-reward to 0 in case of numerical instability
                    if np.isclose(pr, 0, atol=1e-6):
                        pr = 0
                    
                    self.PR[s][t][a]["E"] = pr
                
                    # Set optimal pseudo-reward for current task
                    if curr_task is not None:
                        curr_task.set_optimal_reward(self.PR[s][t][a]["E"])
                    
    def compute_slack_reward(self, t=0):
        if self.slack_reward == 0:
            return 0
        
        if self.gamma < 1:
            cum_discount = ToDoList.get_cum_discount(t)
            return self.slack_reward * ((1 / (1 - self.gamma)) - cum_discount)
        
        return np.PINF

    def get_gamma(self):
        return self.gamma

    def get_highest_negative_reward(self):
        return self.highest_negative_reward

    def get_num_tasks(self):
        return self.num_tasks

    # def get_penalty(self):
    #     return self.penalty
    
    def get_planning_fallacy_const(self):
        return self.planning_fallacy_const
    
    def get_policy(self, s=None, t=None):
        if s is not None:
            if t is not None:
                return self.P[s][t]
            return self.P[s]
        return self.P
    
    def get_q_values(self, s=None, t=None, a=None, t_=None):
        if s is not None:
            if t is not None:
                if a is not None:
                    if t_ is not None:
                        return self.Q[s][t][a][t_]
                    return self.Q[s][t][a]
                return self.Q[s][t]
            return self.Q[s]
        return self.Q

    def get_pseudo_rewards(self, s=None, t=None, a=None, transformed=False):
        """
        Pseudo-rewards are stored as a 3-level dictionaries:
            {(s)tate: {(t)ime: {(a)ction: pseudo-reward}}}
        """
        if transformed:
            if s is not None:
                if t is not None:
                    if a is not None:
                        return self.tPR[s][t][a]
                    return self.tPR[s][t]
                return self.tPR[s]
            return self.tPR

        if s is not None:
            if t is not None:
                if a is not None:
                    return self.PR[s][t][a]
                return self.PR[s][t]
            return self.PR[s]
        return self.PR

    def get_slack_action(self):
        return self.slack_action

    def get_slack_reward(self):
        return self.slack_reward

    def get_tasks(self):
        return self.tasks

    def run_optimal_policy(self, s=None, t=0, t_end=np.PINF, verbose=False):
        """
        
        Args:
            s: Starting state (binary vector of task completion).
            t: Starting time (non-negative integer).
            t_end: Ending time.
            verbose:

        Returns:
            TODO: ...
        """
        # TODO: Write abbreviation description
        
        # If no state was provided, get start state
        if s is None:
            s = tuple(0 for _ in range(self.num_tasks))
            
        # Check whether the state has a valid length
        assert len(s) == self.num_tasks
        
        # Initialize optimal-policy actions list
        opt_P = []

        # Initialize reward placeholder (for printing purposes)
        r = None

        if verbose:
            print(f"\n===== {self.description} =====\n")
            
        while t <= t_end:
            
            # if verbose:
            #     print(self.Q[s][t])

            # Get {actions: {time': reward}} dictionary
            q = self.Q[s][t]
            
            # Get best action and its reward from the dictionary
            a, r_ = ToDoList.max_from_dict(q)
            
            # If next action is not termination
            if a is not None:
                
                # Get next task to "work on"
                goal = self.tasks[a]
                
                # Compute time transition
                t_ = t + goal.get_time_est()

            # Set optimal action for policy in (s)tate and (t)ime
            self.P[s][t] = a

            if verbose:

                # Prepare print if action is termination or slack-off
                print_a = '-' if a is None else a
                print_t_ = '-' if a is None or a == -1 else t_
                print_pr = '-' if a == -1 else self.PR[s][t][a]['E']
    
                if r is not None:
                    print(f"Current reward: {r_ - r} | "
                          # f"Reward difference: {r - r_} | "
                    )
                print(f"Taken action: {print_a} | "
                      f"From time: {t} | "
                      f"To time: {print_t_} | "
                      f"PR: {print_pr} | "
                      f"Future reward: {r_} | "
                      , end=""
                      )

            # If next action is termination or slack-off, terminate
            if a is None or a == -1:
                break

            # Append (action, time') to the optimal-policy actions list
            opt_P.append((a, t_))

            # Move to the next (s)tate
            s = ToDoList.exec_action(s, a)
            
            # Move to the next time step
            t = t_

            # Store reward for printing purposes
            r = r_
    
        if verbose:
            print()
    
        return opt_P, t

    # def scale_rewards(self, min_value=1, max_value=100, print_values=False):
    #     """
    #     Linear transform we might want to use with Complice
    #     """
    #     dict_values = np.asarray([*self.PR.values()])
    #     minimum = np.min(dict_values)
    #     ptp = np.ptp(dict_values)
    #     for trans in self.PR:
    #         self.tPR[trans] = \
    #             max_value * (self.PR[trans] - minimum)/(ptp)

    def set_gamma(self, gamma):
        self.gamma = gamma
        
    def set_slack_reward(self, slack_reward):
        self.slack_action.set_reward(slack_reward)
        self.slack_reward = slack_reward

    def solve(self, start_time=0, verbose=False):
        
        def solve_next_tasks(curr_state, mode, verbose=False):
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

            idx_deadlines = curr_state["idx_deadlines"]
            idx_time_est = curr_state["idx_time_est"]
            
            idx = None
            next_task = None
            queue = None
            task_idx = None

            if mode == "deadline":
                idx = idx_deadlines
                queue = self.sorted_tasks_by_deadlines
            if mode == "time_est":
                idx = idx_time_est
                queue = self.sorted_tasks_by_time_est
            
            # Get next uncompleted task (if one exists)
            while idx < len(queue):
                task = queue[idx]
                task_idx = task.get_idx()

                # If the next task in the queue is completed
                if s[task_idx]:
                    idx += 1
                else:
                    next_task = task
                    break

            if verbose:
                print(
                    f"\nCurrent task-level state: {s} | "
                    f"Current time: {t} | "
                    f"Task index: {task_idx if task_idx is not None else '-'} | "
                    f"Index deadlines: {idx_deadlines}  | "
                    f"Index time_est: {idx_time_est} | ",
                    end=""
                )

            # Initialize policy entries for (s)tate and ((s)tate, (t)ime))
            self.P.setdefault(s, dict())
            self.P[s].setdefault(t, dict())

            # Initialize Q-function entries for (s)tate and ((s)tate, (t)ime))
            self.Q.setdefault(s, dict())
            self.Q[s].setdefault(t, dict())
            
            # Initialize reward entries for (s)tate and ((s)tate, (t)ime))
            self.R.setdefault(s, dict())
            self.R[s].setdefault(t, dict())

            if next_task is not None:
    
                """ Add absorbing terminal state """
                self.Q[s].setdefault(np.PINF, dict())
                self.Q[s][np.PINF].setdefault(None, dict())
                self.Q[s][np.PINF][None]["E"] = 0
    
                self.R[s].setdefault(np.PINF, dict())
                self.R[s][np.PINF].setdefault(None, dict())
                self.R[s][np.PINF][None]["E"] = 0
    
                # Set action to be the index of the next task
                a = next_task.get_idx()
                
                # Initialize Q-value for (a)ction in (s)tate and (t)ime
                self.Q[s][t].setdefault(a, dict())
                
                # Initialize reward for (a)ction in (s)tate and (t)ime
                self.R[s][t].setdefault(a, dict())

                # Initialize expected value for (a)ction
                self.Q[s][t][a].setdefault("E", 0)

                # Get next state by taking (a)ction in (s)tate
                s_ = ToDoList.exec_action(s, a)

                # Initialize Q-values for (s)tate'
                self.Q.setdefault(s_, dict())

                # Get the latest deadline time of the next tasks
                task_deadline = next_task.get_latest_deadline_time()

                # Get time transitions of the next state
                time_transitions = next_task.get_time_transition_prob().items()
                
                if verbose:
                    print(f"Time estimates:"
                          f"{next_task.get_time_transition_prob().items()}")
                    
                # TODO: Remove this (!)
                # task_time_est = next_task.get_time_est()
                # time_ests = [task_time_est, task_time_est+1]
                # probs = [0.50, 0.50]
                
                # print(time_transitions)
                
                # TODO: Trial to fix computing pseudo-rewards...
                mean_time_est = next_task.get_time_est()
                mean_t_ = t + mean_time_est
                
                for time_est, prob_t_ in time_transitions:
                # for time_est, prob_t_ in zip(time_ests, probs):
                
                    # Make time transition
                    t_ = t + time_est

                    # Initialize Q-values for (t)ime'
                    self.Q[s_].setdefault(t_, dict())

                    # Increase total-computations counter
                    self.total_computations += 1

                    # Get cumulative discount w.r.t. task duration
                    cum_discount = ToDoList.get_cum_discount(time_est)
                    
                    # Calculate total loss for next action (immediate "reward")
                    r = next_task.get_total_loss(discount=cum_discount)
                    
                    if verbose:
                        print(f"Current reward {r} | ", end="")
                    
                    # If the transition to the next (t)ime' is already computed
                    if t_ in self.Q[s][t][a].keys():

                        if verbose:
                            print(f"Transition (s, t, a, t') {(s, t, a, t_)} "
                                  f"already computed.")
    
                        # Increase already-computed-pruning counter
                        self.already_computed_pruning += 1

                    else:
    
                        if verbose:
                            print()
    
                        # Initialize expected reward for (state, time, action)
                        self.R[s][t][a].setdefault("E", 0)

                        # Initialize Q value for (state, time, action, time')
                        self.Q[s][t][a][t_] = None

                        # Generate next task-level state
                        next_state = deepcopy(curr_state)
                        next_state["s"] = s_
                        next_state["t"] = t_
                        
                        # Add deadline to the missed deadlines if not attained
                        if self.check_missed_task_deadline(task_deadline, t_):
                            # TODO: Change the way penalties are computed (!)
                            total_penalty = next_task.get_unit_penalty() * \
                                            (t_ - task_deadline)
                            
                            next_state["missed_deadlines"].append(task_deadline)
                            next_state["penalty_factor"] += total_penalty
                            
                        if mode == "deadline":
                            next_state["idx_deadlines"] = idx + 1
                            
                            solve_next_tasks(next_state, mode="time_est", verbose=verbose)
                            solve_next_tasks(next_state, mode="deadline", verbose=verbose)
                        
                        elif mode == "time_est":
                            next_state["idx_time_est"] = idx + 1
                            
                            solve_next_tasks(next_state, mode="time_est", verbose=verbose)
                            solve_next_tasks(next_state, mode="deadline", verbose=verbose)
                            
                        else:
                            raise NotImplementedError(f"Mode {mode} not implemented!")

                        # Store (r)eward for (state, time, action, time')
                        self.R[s][t][a][t_] = r
    
                        # Update expected reward for (state, time, action)
                        self.R[s][t][a]["E"] += prob_t_ * r
    
                        # Get best (a)ction and its (r)eward in (state', time')
                        a_, r_ = ToDoList.max_from_dict(self.Q[s_][t_])
    
                        # Store policy for the next (state, time) pair
                        self.P[s_][t_] = a_
                        
                        """ Compute total reward for the current state-time action
                            Immediate + Expected future reward """
                        
                        # If next best action is a terminal state
                        # Single time-step discount (MDP)
                        # if a_ is None:
                        #     total_reward = r + r_
                        #
                        # Otherwise
                        # else:
                        #     Single time-step discount (MDP)
                        #     total_reward = r + self.gamma * r_
                        
                        # Multiple time-step discount (SMDP)
                        # print(s, t, a, time_est, ToDoList.get_discount(time_est))  # TODO: Remove this line.
                        total_reward = r + ToDoList.get_discount(time_est) * r_
    
                        # If total reward is negative, compare it with the highest
                        # negative reward and substitute if higher
                        if total_reward < 0:
                            self.highest_negative_reward = \
                                max(self.highest_negative_reward, total_reward)
                            
                        # Store Q value for (state, time, action, time')
                        self.Q[s][t][a][t_] = total_reward
                        
                        # Add contribution to the expected value of taking (a)ction
                        self.Q[s][t][a]["E"] += prob_t_ * total_reward
                        
            # ===== Terminal state =====
            else:
                if verbose:
                    print(f"(s, t) {(s, t)} is a terminal state.")

                # Get discount value
                # discount_t = ToDoList.get_discount(t)
                
                # Add goal deadline to the missed deadlines if not attained
                # if t > self.latest_deadline_time:
                #     curr_state["missed_deadlines"].append(
                #         self.latest_deadline_time
                #     )
                
                # Compute penalties for reaching terminal state by this path
                # penalty = self.compute_penalties(
                #     curr_state["missed_deadlines"], t
                # )
                
                # Get total penalty factor
                beta = curr_state["penalty_factor"]
                
                # Compute reward
                # term_value = self.get_reward(t, discount=discount_t) + penalty
                # term_value = self.get_reward(beta=beta, t=t, discount=discount_t)
                term_value = self.get_reward(beta=beta, t=t, discount=1.)

                # Initialize dictionary of Q values for (s)tate and (t)ime
                self.Q.setdefault(s, dict())
                self.Q[s].setdefault(t, dict())
                self.Q[s][t].setdefault(None, dict())
                
                # Compute reward for reaching terminal state s in time t
                self.R[s][t].setdefault(None, dict())
                self.R[s][t][None][np.PINF] = 0
                self.R[s][t][None]["E"] = 0

                # Store (r)eward for (state, time, action, time')
                # self.R[s][t][None][t_] = r

                # Update expected reward for (state, time, action)
                # self.R[s][t][a]["E"] += prob_t_ * r

                # If there is already an assigned value for termination,
                # store the value that maximizes the Q value
                if "E" in self.Q[s][t][None].keys():
    
                    self.Q[s][t][None]["E"] = \
                        max(self.Q[s][t][None]["E"], term_value)
                    
                # Otherwise, initialize it with the current Q value
                else:
                    self.Q[s][t][None]["E"] = term_value
            
        # Initialize starting state and time
        s = tuple(0 for _ in range(self.num_tasks))
        t = start_time
        
        # Initialize state
        curr_state = {
            "s": s,
            "t": t,
            "idx_deadlines": 0,
            "idx_time_est":  0,
            "missed_deadlines": deque(),
            "penalty_factor": 0.,
        }

        # Take next action to be from the task list sorted w.r.t. time estimates
        solve_next_tasks(curr_state, mode="time_est", verbose=verbose)
        
        # Take next action to be from the task list sorted w.r.t. deadlines
        solve_next_tasks(curr_state, mode="deadline", verbose=verbose)
        
        # Get optimal action and value for the starting state and time
        a, r = ToDoList.max_from_dict(self.Q[s][t])

        '''
        Revise optimal policy
        '''
        # TODO: Revise optimal policy by incorporating the slack-off action
        #     - For each step, check whether the slack-off action is better than
        #       working on any other goal. If that is the case, take the
        #       slack-off action and terminate.
        #     - NOTE: This makes sense only if gamma < 1. Otherwise, the
        #             slack-off action is always the best one since it brings
        #             infinite positive reward.
        #     - Done in the run_optimal_policy method!

        # Store policy for the next (state, time) pair
        self.P[s][t] = a
        
        # Return optimal (P)olicy, (Q)-values, (R)ewards
        return {
            "P": self.P,
            "Q": self.Q,
            "R": self.R,
            "s": s,
            "t": t,
            "a": a,
            "r": r
        }

    # def transform_pseudo_rewards(self, print_values=False):
    #     """
    #     TODO: Understand what the method REALLY does...
    #
    #     applies linear transformation to PRs to PR'
    #
    #     linearly transforms PR to PR' such that:
    #         - PR' > 0 for all optimal actions
    #         - PR' <= 0 for all suboptimal actions
    #     """
    #     # Calculate the 2 highest pseudo-rewards
    #     highest = -float('inf')
    #     sec_highest = -float('inf')
    #
    #     for trans in self.pseudo_rewards:
    #         pr = self.pseudo_rewards[trans]
    #         if pr > highest:
    #             sec_highest = highest
    #             highest = pr
    #         elif sec_highest < pr < highest:
    #             sec_highest = pr
    #
    #     # TODO: Understand this...
    #     alpha = (highest + sec_highest) / 2
    #     beta = 1
    #     if alpha <= 1.0:
    #         beta = 10
    #
    #     # TODO: Why (alpha + pr) * beta?! Shouldn't it be (alpha + pr * beta)!?
    #     for trans in self.pseudo_rewards:
    #         self.transformed_pseudo_rewards[trans] = \
    #             (alpha + self.pseudo_rewards[trans]) * beta
    #
    #     if print_values:
    #         print(f'1st highest: {highest}')
    #         print(f'2nd highest: {sec_highest}')
    #         print(f'Alpha: {alpha}')

    
class ToDoList:
    
    # Class attributes
    DISCOUNTS = None
    CUM_DISCOUNTS = None

    def __init__(self, goals, end_time=np.PINF, gamma=1.0, slack_reward=0,
                 start_time=0):
        """

        Args:
            goals: [Goal]
            end_time: End time of the MDP (i.e. horizon)
            gamma: Discount factor
            slack_reward: Unit-time reward for slack-off action.
            start_time:  Starting time of the MDP
        """
        
        self.goals = goals
        self.end_time = end_time
        self.gamma = gamma
        self.slack_reward = slack_reward
        self.start_time = start_time

        # Slack-off action
        self.slack_action = Task("__SLACK-OFF__", deadline=np.PINF,
                                 reward=self.slack_reward, time_est=1)
        
        # Set number of goals
        self.num_goals = len(self.goals)

        # Calculate total time estimate of the to-do list
        self.total_time_est = 0
        
        # Add goals to the to-do list
        self.add_goals(self.goals)
        
        # "Cut" horizon in order to reduce the number of computations
        self.end_time = min(self.end_time, self.total_time_est)
        
        # Generate discounts | TODO: Implement epsilon (!)
        ToDoList.generate_discounts(epsilon=0., gamma=self.gamma,
                                    horizon=self.end_time, verbose=False)
        
        # Initialize policy, Q-value function and pseudo-rewards
        self.P = dict()  # Optimal policy {state: action}
        self.Q = dict()  # Action-value function {state: {action: value}}
        self.PR = dict()  # Pseudo-rewards {(s, t, a): PR(s, t, a)}
        self.tPR = dict()  # Transformed PRs {(s, t, a): tPR(s, t, a)}
        
        self.F = dict()
        self.R = dict()
        self.R_ = dict()

        # Initialize computation counters
        self.small_reward_pruning = 0
        self.already_computed_pruning = 0
        self.total_computations = 0
        
        # Initialize highest negative reward
        self.highest_negative_reward = np.NINF
        
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

    @classmethod
    def exec_action(cls, s, a):
        s_ = list(s)
        s_[a] = 1
        return tuple(s_)

    @classmethod
    def max_from_dict(cls, d: dict):
        max_a = None
        max_r = np.NINF
        
        for a in d.keys():
            r = d[a]["E"]
            if max_r <= r:
                max_a = a
                max_r = r
    
        return max_a, max_r

    def add_goals(self, goals):
        for goal in goals:
        
            # Increase total time estimate of the to-do list
            self.total_time_est += goal.get_time_est()
        
            # Set discount parameter for goal if not defined
            if goal.get_gamma() is None:
                goal.set_gamma(self.gamma)
        
            # Set slack reward for goal's slack-off action if not defined
            if goal.get_slack_reward() is None:
                goal.set_slack_reward(self.slack_reward)

    def compute_pseudo_rewards(self, start_time=0, loc=0., scale=1.):
        """
        Computes pseudo-rewards.

        TODO: Add transformation here.
        """
        standardizing_reward = self.highest_negative_reward
        if self.highest_negative_reward == np.NINF:
            standardizing_reward = 0
            
        # Get time-shift discount
        discount = ToDoList.get_discount(start_time)

        for s in self.Q.keys():
        
            self.PR.setdefault(s, dict())
            self.F.setdefault(s, dict())
        
            for t in self.Q[s].keys():
    
                self.PR[s].setdefault(t, dict())
                self.F[s].setdefault(t, dict())
    
                """ Incorporate slack action for (state, time) """
                # Add slack reward to the dictionary
                slack_reward = self.compute_slack_reward(0)
    
                # Add slack reward in the Q-values
                self.Q[s][t].setdefault(-1, dict())
                self.Q[s][t][-1]["E"] = slack_reward
    
                # Add slack reward in the rewards
                self.R[s][t].setdefault(-1, dict())
                self.R[s][t][-1]["E"] = slack_reward
    
                # The best possible (a)ction and (q)-value in state s
                best_a, best_q = ToDoList.max_from_dict(self.Q[s][t])
                best_q *= discount
                
                # Store optimal policy
                self.P[s][t] = best_a
                
                for a in self.Q[s][t].keys():
                
                    self.PR[s][t].setdefault(a, dict())
                    self.F[s][t].setdefault(a, dict())
                
                    if a is None or a == -1:
    
                        # Set pseudo-reward of terminal state to 0 [Ng, 1999]
                        self.PR[s][t][a].setdefault("E", 0)
                        self.F[s][t][a].setdefault("E", 0)
                        self.R[s][t][a].setdefault("E", 0)
                        
                        # TODO: Comment this part
                        s_ = s
                        t_ = np.PINF
                        gamma = 1.

                        if a == -1:
                            curr_goal = self.slack_action
                        else:
                            curr_goal = None

                        a_ = None
                        q_ = 0

                    else:
                        # Get current Task object
                        curr_goal = self.goals[a]
                    
                        # Move to the next state
                        s_ = ToDoList.exec_action(s, a)
                        
                        # Get expected goal time estimate
                        time_est = curr_goal.get_time_est()
                    
                        # Make time transition
                        t_ = t + time_est

                        # Get gamma for the next transition
                        gamma = ToDoList.get_discount(time_est)

                        # Get optimal action in the next state
                        a_, q_ = ToDoList.max_from_dict(self.Q[s_][t_])

                    # self.R.setdefault(s_, dict())
                    self.R[s_].setdefault(t_, dict())
                    self.R[s_][t_].setdefault(a_, dict())

                    # Expected Q-value of the next state
                    q_ *= discount * gamma
                    # q_ = self.Q[s_][t_][a_]["E"] * discount * gamma
                    
                    # Expected Q-value of the current state
                    # q = self.Q[s][t][a]["E"] * discount
                    
                    # Compute value of the reward-shaping function
                    # f = q_ - q
                    f = q_ - best_q
                    
                    # Standardize rewards s.t. negative rewards <= 0
                    # f -= standardizing_reward * discount

                    # Make affine transformation of the reward-shaping function
                    f = scale * f + loc

                    # Get expected reward for (state, time, action)
                    r = self.R[s][t][a]["E"]

                    # Store reward-shaping value
                    self.F[s][t][a]["E"] = f

                    # Calculate pseudo-reward
                    pr = f + r

                    # Set pseudo-reward to 0 in case of numerical instability
                    if np.isclose(pr, 0, atol=1e-6):
                        pr = 0

                    self.PR[s][t][a]["E"] = pr

                    # Set optimal pseudo-reward for current goal
                    if curr_goal is not None:
                        curr_goal.set_optimal_reward(self.PR[s][t][a]["E"])

    def compute_slack_reward(self, t=0):
        if self.slack_reward == 0:
            return 0
        
        if self.gamma < 1:
            cum_discount = ToDoList.get_cum_discount(t)
            return self.slack_reward * ((1 / (1 - self.gamma)) - cum_discount)
        
        return np.PINF

    def get_end_time(self):
        return self.end_time

    def get_gamma(self):
        return self.gamma
    
    def get_highest_negative_reward(self):
        return self.highest_negative_reward
    
    def get_goals(self):
        return self.goals
    
    def get_num_goals(self):
        return self.num_goals

    def get_optimal_policy(self, state=None):
        """
        Returns the mapping of state to the optimal policy at that state
        """
        if state is not None:
            return self.P[state]
        return self.P

    def get_q_values(self, s=None, a=None):
        if s is not None:
            if a is not None:
                return self.Q[s][a]
            return self.Q[s]
        return self.Q

    def get_pseudo_rewards(self, s=None, t=None, a=None, transformed=False):
        """
        Pseudo-rewards are stored as a 3-level dictionaries:
            {(s)tate: {(t)ime: {(a)ction: pseudo-reward}}}
        """
        if transformed:
            if s is not None:
                if t is not None:
                    if a is not None:
                        return self.tPR[s][t][a]
                    return self.tPR[s][t]
                return self.tPR[s]
            return self.tPR

        if s is not None:
            if t is not None:
                if a is not None:
                    return self.PR[s][t][a]
                return self.PR[s][t]
            return self.PR[s]
        return self.PR

    def get_start_time(self):
        return self.start_time
    
    def get_slack_action(self):
        return self.slack_action

    def get_slack_reward(self):
        return self.slack_reward

    def run_optimal_policy(self, s=None, t=0,
                           run_goal_policy=True, verbose=False):
        # TODO: Write abbreviation description
        
        if s is None:
            s = tuple(0 for _ in range(self.num_goals))
        assert len(s) == self.num_goals
        
        # Initialize optimal-policy actions list
        opt_P = []
        
        # Initialize reward placeholder (for printing purposes)
        r = None
        
        if verbose:
            print("\n===== Goal-level policy =====\n")
            
        while True:
            
            # Add slack reward to the dictionary
            slack_reward = self.compute_slack_reward(0)  # TODO: t or 0 (?)
            
            self.Q[s][t].setdefault(-1, dict())
            self.Q[s][t][-1]["E"] = slack_reward
    
            # if verbose:
            #     print(self.Q[(s, t)])

            # Get {actions: {time': reward}} dictionary
            q = self.Q[s][t]
            
            # Get best action and its reward from the dictionary
            a, r_ = ToDoList.max_from_dict(q)
            
            # If next action is not termination
            if a is not None:
                
                # Get next goal to "work on"
                goal = self.goals[a]
                
                # Compute time transition
                t_ = t + goal.get_time_est()

            # Set optimal action for policy in (s)tate and (t)ime
            self.P[s][t] = a
            
            if verbose:
                
                # Prepare print if action is termination or slack-off
                print_a = '-' if a is None else a
                print_t_ = '-' if a is None or a == -1 else t_
                print_pr = '-' if a == -1 else self.PR[s][t][a]['E']
                
                if r is not None:
                    print(f"Q: {r} | "
                #           f"Reward difference: {r - r_} | "
                    )
                print(f"Taken action: {print_a} | "
                      f"From time: {t} | "
                      f"To time: {print_t_} | "
                      f"F: {print_pr} | "
                      , end=""
                )

            # If next action is termination or slack-off, terminate
            if a is None or a == -1:
                break
                
            # Append (action, time') to the optimal-policy actions list
            opt_P.append((a, t_))

            # Move to the next (s)tate
            s = ToDoList.exec_action(s, a)
            
            # Move to the next time step
            t = t_
            
            # Store reward for printing purposes
            r = r_

        if verbose:
            print("\n")
            print(opt_P)

        """
        Run task-level optimal policy in the order as computed by the goal-level
        optimal policy
        """
        if run_goal_policy:
            
            # Set index of the list of optimal action
            idx = 0
            
            # Set initial time
            t = 0
            
            while idx < len(opt_P):
                # Get action and next time step from the optimal-policy actions list
                a, t_end = opt_P[idx]
                
                # Get next goal
                goal = self.goals[a]
                
                # Compute pseudo-rewards for the next goal
                goal.compute_pseudo_rewards(start_time=t, loc=0, scale=1)
                
                # Run task-level optimal policy for the next goal
                st_, t = goal.run_optimal_policy(t=t, t_end=t_end, verbose=verbose)
                
                idx += 1
            
        return opt_P

    def set_gamma(self, gamma):
        assert 0 < gamma <= 1
        self.gamma = gamma

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

    def set_slack_reward(self, slack_reward):
        self.slack_action.set_reward(slack_reward)
        self.slack_reward = slack_reward

    def solve(self, start_time=0, verbose=False):
    
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
                    f"Current time: {t} | "
                    , end=""
                )
                
            # Initialize next goal
            next_goal = None

            # Initialize policy entries for (s)tate and ((s)tate, (t)ime))
            self.P.setdefault(s, dict())
            self.P[s].setdefault(t, dict())

            # Initialize Q-function entry for the (state, time) pair
            self.Q.setdefault(s, dict())
            self.Q[s].setdefault(t, dict())

            # Initialize reward entries for (s)tate and ((s)tate, (t)ime))
            self.R.setdefault(s, dict())
            self.R[s].setdefault(t, dict())

            """ Add absorbing terminal state """
            self.Q[s].setdefault(np.PINF, dict())
            self.Q[s][np.PINF].setdefault(None, dict())
            self.Q[s][np.PINF][None]["E"] = 0

            self.R[s].setdefault(np.PINF, dict())
            self.R[s][np.PINF].setdefault(None, dict())
            self.R[s][np.PINF][None]["E"] = 0

            # Find the next uncompleted goal
            for goal_idx in range(self.num_goals):
                
                # If the goal with index goal_idx is not completed
                if s[goal_idx] == 0:
    
                    # Increase total-computations counter
                    self.total_computations += 1
    
                    # Set action to be the corresponding goal index
                    a = goal_idx

                    # Initialize Q-value for (a)ction
                    self.Q[s][t].setdefault(a, dict())

                    # Initialize expected value for (a)ction
                    self.Q[s][t][a]["E"] = 0

                    # Generate next state
                    s_ = ToDoList.exec_action(s, a)

                    # Get next Goal object
                    next_goal = self.goals[goal_idx]
                    
                    # Get goal's time estimate
                    time_est = next_goal.get_time_est()

                    # Move for "expected goal time estimate" units in the future
                    t_ = t + time_est
    
                    # Calculate total reward for next action
                    result = next_goal.solve(start_time=t, verbose=verbose)
                    r = result["r"]
                    
                    # Initialize entry for (state, time, action)
                    self.R[s][t].setdefault(a, dict())
                    self.R[s][t][a].setdefault("E", 0)
                    
                    if verbose:
                        print(f"Current reward {r} | ", end="")
    
                    # Initialize Q-values for (s)tate' and (t)ime'
                    self.Q.setdefault(s_, dict())
                    self.Q[s_].setdefault(t_, dict())

                    # The computation has already been done --> Prune!
                    if t_ in self.Q[s][t][a].keys():
        
                        # Increase already-computed-pruning counter
                        self.already_computed_pruning += 1

                        if verbose:
                            print(f"Transition (s, t, a, t') {(s, t, a, t_)} "
                                  f"already computed.")
        
                    # Explore the next goal-level state
                    else:
    
                        # Store reward for (state, time, action, time')
                        self.R[s][t][a][t_] = r
                        
                        # Update expected reward for (state, time, action)
                        self.R[s][t][a]["E"] += r
                        
                        # Initialize key
                        self.Q[s][t][a][t_] = None
                        
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
                        
                        # Compute total reward for the current state-time action as
                        # immediate + (discounted) expected future reward
                        # - Single time-step discount (MDP)
                        # total_reward = r + self.gamma ** next_goal.num_tasks * r_
                        
                        # - Multiple time-step discount (SMDP)
                        total_reward = r + self.get_discount(time_est) * r_
                        
                        # If total reward is negative, compare it with the highest
                        # negative reward and substitute if higher
                        if total_reward < 0:
                            self.highest_negative_reward =\
                                max(self.highest_negative_reward, total_reward)
    
                        # Store Q-value for the current (state, time, action, time')
                        self.Q[s][t][a][t_] = total_reward
    
                        # TODO: Probability of transition to next state
                        prob = 1
    
                        # Store Q-value for taking (a)ction in ((s)tate, (t)ime)
                        self.Q[s][t][a][t_] = total_reward
                        
                        # Add more values to the expected value
                        self.Q[s][t][a]["E"] += prob * total_reward
                        
            # ===== Terminal state =====
            if next_goal is None:
                
                # TODO: Compute total penalty for missing goal deadlines (?!)

                # Add slack reward in the Q-values
                self.Q[s][t].setdefault(-1, dict())
                self.Q[s][t][-1]["E"] = 0

                # Initialize dictionary for the terminal state Q-value
                self.Q[s][t].setdefault(None, dict())
                
                # There is no (additional) reward for completing all goals
                self.Q[s][t][None]["E"] = 0

                # Compute reward for reaching terminal state s in time t
                self.R[s][t].setdefault(None, dict())
                self.R[s][t][None][np.PINF] = 0
                self.R[s][t][None]["E"] = 0

        # Iterate procedure
        s = tuple(0 for _ in range(self.num_goals))
        t = start_time
        
        curr_state = {
            "s": s,
            "t": t
        }

        # Start iterating
        solve_next_goals(curr_state, verbose=verbose)
        
        # Get best (a)ction in the start state and its corresponding (r)eward
        a, r = ToDoList.max_from_dict(self.Q[s][t])
        
        # TODO: Revise optimal policy by incorporating the slack-off action
        #     - For each step, check whether the slack-off action is better than
        #       working on any other goal. If that is the case, take the
        #       slack-off action and terminate.
        #     - NOTE: This makes sense only if gamma < 1. Otherwise, the
        #             slack-off action is always the best one since it brings
        #             infinite positive reward.
        
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
