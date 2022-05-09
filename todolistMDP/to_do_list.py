import numpy as np
import time
#from toolz import memoize, compose
from collections import deque
from copy import deepcopy
from todolistMDP.zero_trunc_poisson import get_binned_dist
from pprint import pprint
import itertools


class Item:

    def __init__(self, description, completed=False, deadline=None,
                 deadline_datetime=None, item_id=None, children=None, value=None,
                 parent_item=None, essential=False, importance=0, time_est=0, today=None,
                 intrinsic_reward=0, num_bins=1):
        """

        :param description: name of Item
        :param completed: Flag if Item is completed. In case of sub-Items: An item is said to be complete if all its
                          essential super-ordinate items are completed.
        :param deadline: deadline of Item
        :param deadline_datetime: deadline of Item in datetime structure
        :param item_id: Item ID
        :param children: Immediate sub items to be considered
        :param value: Value for completing of super-ordinate item
        :param parent_item: Parent of current item
        :param essential: Flag if item is set to be essential to complete super-ordinate item.
                          All goal items are automatically considered essential
        :param importance: Degree of importance to completion of super-ordinate item. In scale of 1.
        :param time_est: Time estimate to complete task
        :param today: Flag if item is marked to be completed today
        :param intrinsic_reward: Intrinsic reward for completing item
        """

        # Compulsory parameters
        self.description = description

        # Optional parameters
        self.completed = completed
        self.deadline = deadline
        self.deadline_datetime = deadline_datetime
        self.parent_item = parent_item
        self.time_est = time_est
        self.today = today
        self.intrinsic_reward = intrinsic_reward
        self.essential = essential
        self.importance = importance
        self.value = value
        self.num_bins = num_bins
        # Initialize index
        self.idx = None

        # Initialize 0-th state
        self.start_state = tuple()
        # Initialize list of sub-items
        self.children = deque()
        self.done_children = deque()
        self.today_children = set()
        # Initialize number of children
        self.num_children = 0

        self.item_id = item_id
        if self.item_id is None:
            self.item_id = self.description

        # Initialize time transitions
        self.time_transitions = {
            self.time_est: 1.
        }

        # Initialize expected reward
        self.expected_rewards = []
        self.immediate_rewards = []
        self.optimal_reward = 0

        # Initialize dictionary of maximum future Q-values
        self.future_q = {
            None: None
        }

        # Add items on the next level/depth
        if children is not None:
            self.add_children(children)
            self.time_est = sum([child.time_est for child in self.children])

        # print(f'Item: {self.description} Time Estimate: {self.time_est}')


    def add_children(self, children):
        """

        :param children:
        :return:
        """

        appended_state = list(0 for _ in range(len(children)))
        self.start_state = appended_state

        self.children = deque()
        self.done_children = deque()
        self.num_children = 0
        self.today_children = set()

        self.importance = 0
        self.intrinsic_reward = 0
        for idx, child in enumerate(children):
            # Shift children index
            idx += self.num_children

            # Set item index
            child.set_idx(idx)
            self.importance += child.importance
            self.intrinsic_reward += child.intrinsic_reward
            # Add child that has to be executed today
            if child.is_today() and not child.is_completed():
                self.today_children.add(child)

            # Set child as completed in the start state
            if child.is_completed():
                self.start_state[idx] = 1
                self.done_children.append(child)

            # Add child
            self.children.append(child)

            # Set goal as a parent item
            child.set_parent_item(self)

        # Convert start state from list to tuple
        self.start_state = tuple(self.start_state)

        # Set number of children_
        self.num_children = len(self.start_state)

    def compute_binning(self, num_bins=None):
        binned_dist = get_binned_dist(mu=self.time_est, num_bins=num_bins)

        bin_means = binned_dist["bin_means"]
        bin_probs = binned_dist["bin_probs"]

        self.time_transitions = dict()

        for i in range(len(bin_means)):
            mean = int(np.ceil(bin_means[i]))
            prob = bin_probs[i]

            self.time_transitions[mean] = prob

    def set_idx(self, idx):
        self.idx = idx

    def get_idx(self):
        return self.description

    def is_completed(self):
        return self.completed

    def set_parent_item(self, parent_item):
        self.parent_item = parent_item

    def is_today(self):
        return self.today

    def get_optimal_reward(self):
        return self.optimal_reward

    def get_deadline(self):
        return self.deadline

    def is_deadline_missed(self, t):
        return t > self.deadline

    def get_time_transitions(self, t=None):
        if t is None:
            return self.time_transitions
        return self.time_transitions[t]

    def set_expected_reward(self, expected_reward):
        # print(f'In set_expected_reward {self.description}')
        self.expected_rewards.append(expected_reward)

    def set_immediate_reward(self, immediate_reward):
        # print(f'In set_expected_reward {self.description}')
        self.immediate_rewards.append(immediate_reward)

    def get_future_q(self, t=None):
        if t is not None:
            return self.future_q[t]
        return self.future_q

    def set_future_q(self, t, value, compare=False):
        # Set future Q-value
        self.future_q.setdefault(t, value)

        if compare:
            self.future_q[t] = max(self.future_q[t], value)

    def get_expected_reward(self):
        return np.mean(self.expected_rewards)

    def get_immediate_reward(self):
        return np.mean(self.immediate_rewards)

    def get_intrinsic_reward(self):
        return self.intrinsic_reward

    def get_importance(self):
        return self.importance

    def get_essential(self):
        return self.essential

    def get_time_est(self):
        return self.time_est

    def check_completed(self):
        return self.is_completed()

    def set_optimal_reward(self, optimal_reward):
        self.optimal_reward = optimal_reward

class MDP:
    def __init__(self, items, value=0, end_time=np.PINF, gamma=1.0, loss_rate=0.,
                 num_bins=1, penalty_rate=0., planning_fallacy_const=1.,
                 slack_reward_rate=0, start_time=0):

        self.value = value # Value of solving main goal of MDP
        self.end_time = end_time
        self.gamma = gamma
        self.loss_rate = loss_rate
        self.num_bins = num_bins
        self.penalty_rate = penalty_rate
        self.planning_fallacy_const = planning_fallacy_const
        self.slack_reward_rate = slack_reward_rate
        self.start_time = start_time

        # Initialize list of items
        self.items = deque()

        # Set number of goals
        self.num_items = 0

        # Initialize 0-th state
        self.start_state = tuple()

        # Calculate total time estimate of the to-do list
        self.total_time_est = 0

        self.importance_list = deque()
        self.int_r_list = deque()
        self.essential_list = deque()

        self.discounts = None
        self.cum_discounts = None

        # Add goals to the to-do list
        self.add_items(items)

        # "Cut" horizon in order to reduce the number of computations
        self.end_time = min(self.end_time, self.total_time_est)

        # Generate discounts | TODO: Add epsilon as input parameter (!)
        self.generate_discounts(epsilon=0., gamma=self.gamma,
                                    horizon=self.end_time, verbose=False)

        # Slack-off reward
        self.slack_reward = self.compute_slack_reward()

        # Initialize policy, Q-value function and pseudo-rewards
        self.P = dict()  # Optimal policy {state: action}
        self.Q = dict()  # Action-value function {state: {action: value}}
        self.R = dict()  # Expected rewards
        self.Q_s0 = dict()

        # Initialize computation counters
        self.already_computed_pruning = 0
        self.total_computations = 0

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

    def generate_discounts(self, epsilon=0., gamma=1., horizon=1, verbose=False):
        tic = time.time()

        self.discounts = deque([1.])
        self.cum_discounts = deque([0., 1.])

        for t in range(1, horizon + 1):
            last_power_value = self.discounts[-1] * gamma
            if last_power_value > epsilon:
                self.discounts.append(last_power_value)
                self.cum_discounts.append(
                    self.cum_discounts[-1] + last_power_value)
            else:
                break

        self.discounts = list(self.discounts)
        self.cum_discounts = list(self.cum_discounts)

        toc = time.time()

        if verbose:
            print(f"\nGenerating powers took {toc - tic:.2f} seconds!\n")

        return self.discounts, self.cum_discounts

    @staticmethod
    def compute_total_loss(cum_discount, loss_rate):
        return loss_rate * cum_discount

    def get_cum_discount(self, t):
        n = len(self.cum_discounts)
        cum_discount = self.cum_discounts[min(t, n - 1)]

        if t < 0:
            raise Exception("Negative time value for cumulative discount is "
                            "not allowed!")
        return cum_discount

    def get_discount(self, t):
        n = len(self.discounts)
        discount = self.discounts[min(t, n - 1)]

        if t < 0:
            raise Exception("Negative time value for discount is not allowed!")

        return discount

    def get_policy(self, s=None, t=None):
        if s is not None:
            if t is not None:
                return self.P[s][t]
            return self.P[s]
        return self.P

    def get_optimal_policy(self, state=None):
        """
        Returns the mapping of state to the optimal policy at that state
        """
        if state is not None:
            return self.P[state]
        return self.P

    def compute_slack_reward(self):
        if self.slack_reward_rate == 0:
            return 0

        if self.gamma < 1:
            return self.slack_reward_rate * (1 / (1 - self.gamma))

        return np.PINF

    def get_start_time(self):
        return self.start_time

    def add_items(self, items):

        # Update number of goals
        self.num_items += len(items)

        # Initialize start state
        self.start_state = list(0 for _ in range(self.num_items + 1))

        for idx, item in enumerate(items):

            # Increase total time estimate of the to-do list
            self.total_time_est += item.get_time_est()

            # Set goal index
            item.set_idx(idx)

            # Add item to the list of items
            self.items.append(item)

            # Add to importance list
            self.importance_list.append(item.get_importance())

            # Add to intrinsic reward list
            self.int_r_list.append(item.get_intrinsic_reward())

            # Add to essential list
            self.essential_list.append(item.get_essential())
            if item.check_completed():
                self.start_state[idx] = 1

        # Convert list to tuple
        self.importance_list = tuple(self.importance_list)
        self.int_r_list = tuple(self.int_r_list)
        self.essential_list = tuple(self.essential_list)
        self.start_state = tuple(self.start_state)

    def get_slack_reward(self):
        return self.slack_reward

    def get_reward(self, action, s_=None, value=None, importance_list=None, int_r_list=None, beta=0., discount=1.):
        if s_ is None:
            s_ = self.start_state
        if value is None:
            value = self.value
        if importance_list is None:
            importance_list = self.importance_list
        if int_r_list is None:
            int_r_list = self.int_r_list

        if action == self.num_items:
            return self.slack_reward
        intrinsic_reward = int_r_list[action]
        extrinsic_reward = self.extrinsic_reward(value=value, curr_state=s_[:-1], importance_list=importance_list)
        # print(f'Intrinsic: {intrinsic_reward}, Extrinsic: {extrinsic_reward} Beta: {beta}')
        reward = discount * (intrinsic_reward + (extrinsic_reward / (1 + beta)))
        # print(f'Reward: {reward}')
        return reward

    def extrinsic_reward(self, value=None, curr_state=None, importance_list=None):
        if curr_state is None:
            curr_state = self.start_state
        if importance_list is None:
            importance_list = self.importance_list
        if value is None:
            value = self.value

        sum_importance = 0
        # print(f'Curr_state: {curr_state}')
        # print(f'essential_List: {self.essential_list}')
        for idx, state in enumerate(curr_state):
            # print(idx, state)
            if curr_state[idx] == 0 and self.essential_list[idx] == 1:  # Not completed essential task
                return 0  # Not completed has 0 extrinsic reward
            if curr_state[idx] == 1:
                sum_importance +=  importance_list[idx]
        # print(f'Val: {value}, sum_imp: {sum_importance}, Net: {sum(importance_list)}')
        # If here, that means that all essential tasks have been completed
        return value * (sum_importance / sum(importance_list))

    def solve(self, start_time=None, verbose=False):
        # print("Solving for following items")
        # for item in self.items:
        #     print(f'{item.description}, Imp: {item.importance}, Int: {item.intrinsic_reward}, Ess: {item.essential}')
        # print(f'Value: {self.value}')

        def solve_next_item(curr_state, verbose=False):
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
            # print(f'Called solve_next_item: s: {s}, t: {t}')
            if verbose:
                print(
                    f"Current state: {s} | "
                    f"Current time: {t:>3d} | "
                    , end=""
                )
            # Initialize next item
            next_item = None

            # Initialize policy entries for state and (state, time))
            self.P.setdefault(s, dict())

            # Initialize Q-function entry for the (state, time) pair
            self.Q.setdefault(s, dict())
            self.Q[s].setdefault(t, dict())

            # Initialize reward entries for state and (state, time))
            self.R.setdefault(s, dict())
            self.R[s].setdefault(t, dict())
            # Find the next uncompleted goal
            if s[self.num_items] != 1:
                for item_idx in range(self.num_items + 1):
                    # If the goal with index goal_idx is not completed
                    # print(f'In solve_next_item: s: {s}')
                    # print(f'Item idx: {item_idx}')
                    if s[item_idx] == 0:
                        # print(f'MDP state: {s}, next item: {item_idx}')
                        # Increase total-computations counter
                        self.total_computations += 1

                        # Set action to be the corresponding goal index
                        a = item_idx
                        # print(f'In State {s}, Action considered: { a}')
                        # Generate next state
                        s_ = MDP.exec_action(s, a)  # s_ is a different object to s

                        # Get next child object
                        try:
                            next_item = self.items[item_idx]
                        except:
                            next_item = 'slack' # slack off action 

                        # TODO: Probability of transition to next state
                        prob = 1
                        # print(f's: {s}, s_: {s_}')  # verbose
                        # The computation has already been done --> Prune!
                        if a in self.Q[s][t].keys():

                            # Increase already-computed-pruning counter
                            self.already_computed_pruning += 1

                            if verbose:
                                print(f"Transition (s, t, a, t') {(s, t, a)} "
                                    f"already computed.")

                        # Explore the next item state
                        else:
                            if s_[self.num_items] != 1: 
                                # Initialize expected value for action
                                self.Q[s][t].setdefault(a, 0)

                                # Initialize entry for (state, time, action)
                                self.R[s][t].setdefault(a, 0)
                                # Get deadline time for next item
                                task_deadline = next_item.get_deadline()

                                # Get time transitions of the next state
                                time_transitions = next_item.get_time_transitions().items()
                                exp_task_reward = 0
                                exp_total_reward = 0
                                beta = 0
                                # print(f'Time Transitions; {time_transitions}')
                                for time_est, prob_t_ in time_transitions:

                                    # Increase total-computations counter
                                    self.total_computations += 1

                                    # Make time transition
                                    t_ = t + time_est

                                    # Initialize Q-values for state' and time'
                                    self.Q.setdefault(s_, dict())
                                    self.Q[s_].setdefault(t_, dict())

                                    # Get cumulative discount w.r.t. item duration
                                    cum_discount = self.get_cum_discount(time_est)
                                    # print(f'State: {s}, s_: {s_}' )
                                    # print(f'Time_est: {time_est}, Cum_discount: {cum_discount}, Loss_rate: {self.loss_rate}')
                                    # print(f'Penalty Rate: {self.penalty_rate}"')
                                    # Calculate total loss for next action (immediate "reward")
                                    r = MDP.compute_total_loss(
                                        cum_discount=cum_discount, loss_rate=self.loss_rate
                                    )
                                    # print(f'Just Loss Term: {r}')
                                    # Add deadline to the missed deadlines if not attained
                                    if next_item.is_deadline_missed(t_):
                                        # print(f'Item: {next_item.description} Deadline Missed')
                                        # Compute total penalty for missing item deadline
                                        total_penalty = \
                                            self.penalty_rate * (t_ - task_deadline)

                                        # Update penalty
                                        beta += prob_t_ * total_penalty
                                        # print(f'Beta: {beta}')
                                        # print(prob_t_, total_penalty)
                                    # # Original Code
                                    immediate_reward = self.get_reward(action= a, s_=s_, value=self.value, importance_list=self.importance_list, int_r_list=self.int_r_list,  beta=beta, discount=1)
                                    r += self.get_reward(action= a, s_=s_, value=self.value, importance_list=self.importance_list, int_r_list=self.int_r_list,  beta=beta, discount=1)
                                    # print(f'Adding reward: {self.get_reward(action= a, s_=s_, value=self.value, importance_list=self.importance_list, int_r_list=self.int_r_list,  beta=beta, discount=1)}, Net: {r}')
                                    # #  DEBUG Mode
                                    # if next_item.is_deadline_missed(t_):
                                    #     print(f'State: {s} action: {a} Deadline Missed.')
                                    #     r += 0
                                    # else:
                                    #     r += self.get_reward(action= a, s_ = s_, value=self.value, importance_list=self.importance_list, int_r_list=self.int_r_list,  beta=beta, discount=1)

                                    # Update expected reward for (state, time, action)
                                    exp_task_reward += prob_t_ * r
                                    # print(f'Multiplying with prob_t: {prob_t_}: Exp_task_reward: {exp_task_reward}')

                                    # Generate next  state
                                    state_dict = {
                                        "s": s_,
                                        "t": t_
                                    }

                                    # Explore the next state
                                    solve_next_item(state_dict, verbose=verbose)

                                    # Get best action and its respective for (state', time')
                                    # print(f'For s_: {s_}, t_: {t_}, Qs: {self.Q[s_][t_]}')
                                    a_, r_ = MDP.max_from_dict(self.Q[s_][t_])

                                    # Store policy for the next (state, time) pair
                                    self.P[s_][t_] = a_

                                    # Store future Q-value
                                    next_item.set_future_q(t_, r_, compare=True)

                                    # Compute total reward for the current state-time action as
                                    # immediate + (discounted) expected future reward
                                    # - Single time-step discount (MDP)
                                    # total_reward = r + self.gamma ** next_goal.num_tasks * r_

                                    # - Multiple time-step discount (SMDP)
                                    total_reward = r + self.get_discount(time_est) * r_
                                    # print(f'r: {r} + r_: {r_} * {self.get_discount(time_est)} ')
                                    # print(f'Total Reward: {total_reward}')
                                    exp_total_reward += prob_t_ * total_reward
                                    # print(f'Exp_total_reward: {exp_total_reward}')

                                # TODO: Probability of transition to next state
                                prob = 1

                                # Add more values to the expected value
                                self.Q[s][t][a] += prob * exp_total_reward
                                self.R[s][t][a] += prob * exp_task_reward

                                if s == self.initial_state:
                                    # print(f'Setting Item: {next_item.description} expected reward; {exp_task_reward}')
                                    # print(f'In solve_next_child: state: {s}')
                                    # print(f'{next_child.description}: exp_reward: {exp_task_reward}')
                                    next_item.set_expected_reward(exp_task_reward)
                                    next_item.set_immediate_reward(immediate_reward)

                        # Store initial-state Q-value (if initial time applies)
                        if t == self.start_time:
                            if next_item is 'slack':
                                self.Q_s0[next_item] = self.slack_reward
                            else:
                            # print(f'Storing Q-value of {next_item.description}: {self.Q[s][t][a]}')
                                self.Q_s0[next_item] = self.Q[s][t][a]

            # ===== Terminal state ===== (All children visited)
            if next_item is None:
                # Initialize dictionary for the terminal state Q-value
                self.Q[s][t].setdefault(None, 0)

                # Compute reward for reaching terminal state s in time t
                self.R[s][t].setdefault(None, 0)
            elif next_item is 'slack':
                self.Q[s][t].setdefault(None, self.slack_reward)
                self.R[s][t].setdefault(None, self.slack_reward)

        # Initialize start state & time
        # print(f'self.start_time: {self.start_time}')
        t = self.start_time if start_time is None else start_time
        s = [] #tuple(0 for _ in range(self.num_items))
        for check_item in self.items:
            if check_item.check_completed():
                s.append(1)
            else:
                s.append(0)
        s.append(0) # slack off not selected in the start state
        s = tuple(s)
        # print(f'Initial s: {s}')  # verbose
        self.initial_state = s

        curr_state = {
            "s": s,
            "t": t
        }

        # Start iterating
        solve_next_item(curr_state, verbose=verbose)

        # Get best action in the start state and its corresponding reward
        a, r = MDP.max_from_dict(self.Q[s][t])

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

    def compute_start_state_pseudo_rewards(self, bias=None, scale=None, flag=False):
        '''

        :param bias:
        :param scale:
        :param flag: if True: then only apply scale and bias.
        :return:
        '''
        # print(f'****** In compute_start_state_pseudo_rewards ******')
        # Get list of items
        items = self.items

        # Get goal future Q-values
        item_q = dict()  # {item ID: Q-value for item execution in s[0]}
        next_q = dict()  # {item ID: Q-value after item execution in s[0]}

        # Initialize best Q-value
        best_q = np.NINF

        # Initialize start time
        start_time = self.get_start_time()
        # print("Start Time: {}".format(start_time))

        # Initialize list of incentivized items
        incentivized_items = deque()

        # Initialize total reward
        total_reward = 0

        # Initialize slack reward
        slack_reward = self.get_slack_reward()
        # print(f'slack rewards: {slack_reward}')
        expected = dict()
        for item, q in self.Q_s0.items():


            if best_q <= slack_reward:
                best_q = slack_reward
                # print(f'best_q new: {best_q}')
            # Get item ID
            item_id = item.description
            # Store Q-value for item execution in s[0]
            item_q[item_id] = q

            reward = item.get_expected_reward()
            expected[item_id] = reward
            total_reward += reward

            # Compute Q-value after transition
            next_q[item_id] = q - reward

            # Just to check
            t_ = item.get_time_est()
            future_q = item.get_future_q(start_time + t_)
            # print(f'Item: {item.description} q: {np.round(q,4)}, reward: {np.round(reward, 4)} future_q: {np.round(future_q, 4)}, q - reward: {np.round(q - reward, 4)}')

            # Update best Q-value and best next Q-value
            if best_q <= q:
                best_q = q
                # print(f'best_q newer: {best_q}')

            # Add items to the list of incentivized items (?!)
            incentivized_items.append(item)


        # Initialize minimum pseudo-reward value (bias for linear transformation)
        min_pr = 0

        # Initialize sum of pseudo-rewards
        sum_pr = 0
        # print(f'Init sum_pr: {sum_pr}')
        # Compute untransformed pseudo-rewards
        # print(f'best_q: {best_q}')
        optimal = dict()
        for item in incentivized_items:

            # Get item ID
            item_id = item.description

            # Compute pseudo-reward
            # print(f'Item: {item.description}, next_q: {next_q[item_id]}, best_q: {best_q}')
            pr = next_q[item_id] - best_q

            if np.isclose(pr, 0, atol=1e-6):
                pr = 0

            # Store pseudo-reward
            # print(f'{item.description}: unscaled f*: {pr}')
            item.set_optimal_reward(pr)
            optimal[item_id] = pr
            # Update minimum pseudo-reward
            min_pr = min(min_pr, pr)

            # Update sum of pseudo-rewards
            sum_pr += pr
        # print(f'sum_pr: {sum_pr}')

        # Compute sum of goal values
        # sum_item_values = self.value + sum([item.intrinsic_reward for item in items])  # Takes intrinsic value down
        sum_item_values = self.value
        # print(f'Sum Of all Values: {sum_item_values}')

        if flag:
            # Set value of scaling parameter
            if scale is None:
                # As defined in the report
                scale = 1.10

            # Set value of bias parameter
            if bias is None:
                # Total number of incentivized items
                n = len(incentivized_items)

                # Derive value of the bias term
                bias = (sum_item_values - scale * sum_pr) / n

                # Take total reward into account
                bias -= (total_reward / n)
        else:
            scale = 1
            bias = 0


        # Initialize {item ID: pseudo-reward} dictionary
        id2pr = dict()

        # Sanity check for the sum of pseudo-rewards
        sc_sum_pr = 0
        # print(f'Init sc_sum_pr: {sc_sum_pr}')

        # Perform linear transformation on item pseudo-rewards
        for item in incentivized_items:

            # Get item unique identification
            item_id = item.description

            # Transform pseudo-reward
            pr = f = scale * item.get_optimal_reward() + bias
            # Get expected item reward
            item_reward = item.get_expected_reward()

            # print(f'{item.description}: r: {item_reward}, pr: {pr}, new_pr: {pr + item_reward}')
            # print(f'Scale: {scale} Optimal Reward: {item.get_optimal_reward()}, Bias: {bias}')
            # Add immediate reward to the pseudo-reward
            pr += item_reward

            # Store new (zero) pseudo-reward
            item.set_optimal_reward(pr)

            # If item is not slack action
            if item.get_idx() != -1:
                # Update sanity check for the sum of pseudo-rewards
                sc_sum_pr += item.get_optimal_reward()
            # Store pseudo-reward {item ID: pseudo-reward}
            id2pr[item_id] = pr
            # print(f'Final: {item.description}, {pr}')

            # print(
            #     f"{item.get_description():<70s} | "
            #     # f"{best_next_q:>8.2f} | "
            #     f"max Q*(s', a'): {next_q[item_id]:>8.2f} | "
            #     f"Q*(s, a): {item_q[item_id]:>8.2f} | "
            #     f"V*(s): {best_q:>8.2f} | "
            #     f"f*(s, a): {item.get_optimal_reward():8.2f} | "
            #     f"f*(s, a) + b: {f:8.2f} | "
            #     f"r(s, a, s'): {item.get_expected_reward():>8.2f} | "
            #     f"r'(s, a, s'): {pr:>8.2f}"
            # )



        # print(f"\nTotal sum of pseudo-rewards: {sc_sum_pr:.2f}\n")
        return {
            "incentivized_items": incentivized_items,
            "expected": expected,
            "optimal": optimal,
            "id2pr": id2pr,
            "sc_sum_pr": sc_sum_pr,
            "scale": scale,
            "bias": bias
        }

class MainToDoList:

    def __init__(self, to_do_list_description, end_time=np.PINF, gamma=1.0, loss_rate=0.,
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

        # Initialize complete to_do_list
        self.complete_to_do_list = to_do_list_description
        self.end_time = end_time
        self.gamma = gamma
        self.loss_rate = loss_rate
        self.num_bins = num_bins
        self.penalty_rate = penalty_rate
        self.planning_fallacy_const = planning_fallacy_const
        self.slack_reward_rate = slack_reward_rate
        self.start_time = start_time

        self.Qexact = dict()
        self.Pexact = dict()
        self.Rexact = dict()
        self.discounts = None
        self.cum_discounts = None

      # Add root node
        self.root = Item(description="Root", completed=False, deadline=np.PINF, children=self.complete_to_do_list,
                         essential=True, value=0)

        for goal in self.root.children:
            self.set_parent(goal, self.root)

        # Get tasks
        self.tasks = tuple(self.flatten(self.get_tasks(self.root)))
        self.left_tasks = list(self.tasks)
        for task in self.tasks:
            if task.is_completed():
                self.left_tasks.remove(task)
        self.left_tasks = tuple(self.left_tasks)
        self.num_tasks = len(self.tasks)
        self.sub_goals = tuple(self.get_sub_tasks())
        self.num_sub_goals = len(self.sub_goals)
        # TO DO:
        #   Check the to do list in node.py:
        #   The parent node r_in, importance = sum(children node)
        #   If importance of task is not given, divide remainder importance equally
        #   If essentialness of task is not given, mark it as non-essential
        #   Parent is essential if any child is essential.
        #   Parent is complete if all essential children are complete
        #   If item has no deadline, set super-item's deadline

        # self.update(self.root)

        # Set nodes from the nested information
        self.node_list = list(self.flatten(self.get_nodes(self.root, False)))
        self.node_list.insert(0, self.root)
        self.node_names = list(self.flatten(self.get_nodes(self.root, True)))
        self.node_names.insert(0, self.root.description)
        self.node_names = tuple(self.node_names)
        self.nodes = dict()
        for i, node in enumerate(self.node_list):
            self.nodes[self.node_list[i].description] = self.node_list[i]
        # Set Value dictionary used to store optimal rewards
        value_dict = dict()
        pr_dict = dict()
        expected_dict = dict()
        optimal_dict = dict()
        for node in tuple(self.node_names):
            value_dict[node] = 0
            pr_dict[node] = 0
            expected_dict[node] = 0
            optimal_dict[node] = 0
        self.value_dict = value_dict
        self.pr_dict = pr_dict
        self.expected_dict = expected_dict
        self.optimal_dict = optimal_dict

        self.tree = dict()
        self.tree_recurse([self.root])
        self.task_dict()
        self.goal_dict()

        self.slack_reward = self.compute_slack_reward()
        self.total_time_est = sum([task.time_est for task in self.tasks])
        # "Cut" horizon in order to reduce the number of computations
        self.end_time = min(self.end_time, self.total_time_est)

        # Generate discounts | TODO: Add epsilon as input parameter (!)
        self.generate_discounts(epsilon=0., gamma=self.gamma,
                                horizon=self.end_time)

    @staticmethod
    def flatten(iterable):
        """Recursively iterate lists and tuples.
        """
        for elm in iterable:
            if isinstance(elm, (list, tuple)):
                for relm in MainToDoList.flatten(elm):
                    yield relm
            else:
                yield elm

    @staticmethod
    def recurse(root):
        if len(root.children) == 0:
            return root
        rcg = [MainToDoList.recurse(child) for child in root.children]
        return rcg

    def task_dict(self):
        ind_dict = dict()
        for t in self.tree:
            ind_dict[t] = []
            if len(self.tree[t]) != 0:
                for child in self.tree[t]:
                    op = MainToDoList.recurse(child)
                    if isinstance(op, list):
                        op = MainToDoList.flatten(op)
                        for o in op:
                            ind_dict[t].append(o.description)
                    else:
                        ind_dict[t].append(op.description)
            self.ind_dict = ind_dict

    def goal_dict(self):
        ind_dict = dict()
        for child in self.tree['Root']:
            ind_dict[child.description] = []
            op = MainToDoList.recurse(child)
            if isinstance(op, list):
                op = MainToDoList.flatten(op)
                for o in op:
                    ind_dict[child.description].append(o.description)
            else:
                ind_dict[child.description].append(op.description)
        self.goal_tasks = ind_dict

    def compute_slack_reward(self):
        if self.slack_reward_rate == 0:
            return 0

        if self.gamma < 1:
            return self.slack_reward_rate * (1 / (1 - self.gamma))

        return np.PINF

    def generate_discounts(self, epsilon=0., gamma=1., horizon=1):
        self.discounts = deque([1.])
        self.cum_discounts = deque([0., 1.])

        for t in range(1, horizon + 1):
            last_power_value = self.discounts[-1] * gamma
            if last_power_value > epsilon:
                self.discounts.append(last_power_value)
                self.cum_discounts.append(
                    self.cum_discounts[-1] + last_power_value)
            else:
                break

        self.discounts = list(self.discounts)
        self.cum_discounts = list(self.cum_discounts)

        return self.discounts, self.cum_discounts

    def get_cum_discount(self, t):
        n = len(self.cum_discounts)
        cum_discount = self.cum_discounts[min(t, n - 1)]

        if t < 0:
            raise Exception("Negative time value for cumulative discount is "
                            "not allowed!")
        return cum_discount

    def get_discount(self, t):
        n = len(self.discounts)
        discount = self.discounts[min(t, n - 1)]

        if t < 0:
            raise Exception("Negative time value for discount is not allowed!")

        return discount

    def get_tasks(self, root):
        all_tasks = []
        for goal in root.children:
            rcg = MainToDoList.recurse(goal)
            try:
                rcg = list(itertools.chain(*rcg))
            except:
                rcg = list(itertools.chain(rcg))
            # for rr in rcg:
            all_tasks.extend(rcg)
        return all_tasks

    def get_sub_tasks(self):
        sub_tasks = set()
        for task in self.left_tasks:
            sub_tasks.add(task.parent_item.description)
        return sub_tasks

    def set_parent(self, node, parent):
        node.parent_item = parent
        if len(node.children) != 0:
            for c in node.children:
                self.set_parent(c, node)

    def _render(self, mode='notebook', close=False):
        """
        Renders the environment structute
        """

        if close:
            return
        from graphviz import Digraph

        dot = Digraph()
        for ys in self.tree:
            xxs = []
            for xx in self.tree[ys]:
                xxs.append(xx.description)
            c = 'grey'
            dot.node(str(ys), style='filled', color=c)
            for y in xxs:
                dot.edge(str(ys), str(y))
        return dot

    def get_nodes(self, root, desc=False):
        nodes = []
        for goal in root.children:
            if desc == True:
                nodes.append(goal.description)
            else:
                nodes.append(goal)
            for child in root.children:
                nodes.append(self.get_nodes(child, desc))
        return nodes

    def tree_recurse(self, root_list):
        for root in root_list:
            self.tree[root.description] = root.children
            for goal in root.children:
                self.tree[goal.description] = goal.children
                self.tree_recurse(goal.children)

    def root_update(self, root=None):
        if root is None:
            root = self.root

    def update(self, root=None):  # Not used atm
        if root is None:
            root = self.root
        self.current_pass(self.tasks)

    def current_pass(self, task_list):  # Not used atm Useful template to see how to traverse tree from bottom up
        task_list = list(task_list)
        next_list = []
        while len(task_list) != 0:
            task = task_list[0]
            # print(f'Desc: {task.parent_item.description}, Rew: {task.parent_item.intrinsic_reward}')
            next_list.append(task.parent_item)
            in_sum = 0
            for child in task.parent_item.children:
                print(f'Child: {child.description}, {child.intrinsic_reward}')
                in_sum += child.intrinsic_reward
                task_list.remove(child)
            task.parent_item.intrinsic_reward = in_sum
            # print(f'Updated rew: {task.parent_item.intrinsic_reward}')

        self.current_pass(next_list)

    def solve(self, start_time=None, verbose=False, flag=False):
        # print(f'*********** In MainToDoList solve ***********')
        if start_time is None:
            start_time = self.start_time

        # Only goal value is treated as intrinsic reward. Value of completing root node = 0
        # Solve the small MDPs and go down the tree
        mdps = dict()
        for node in self.tree:
            # print("*********** MDP ***********")
            # print(f'Solving goal as: {self.nodes[node].description}')
            mdp = self.solve_MDP(self.nodes[node], self.tree[node], start_time, verbose, flag=flag)
            if node in self.sub_goals:
                # print(f'Saving Q: {node}')
                mdps[node] = mdp

        self.mdps = mdps
        tasks = np.array(self.left_tasks)
        task_set = set([task.description for task in tasks])
        for key in self.mdps:
            mdp = self.mdps[key]
            # print(f'Start_state: {mdp.start_state}')  # verbose
            # print(mdp.Q[mdp.start_state])  # verbose
            item_set = set([item.description for item in mdp.items])
            if item_set & task_set:
                for i, item in enumerate(mdp.items):
                    # print(f'{i}: {item.description}')  
                    if item.description in task_set:
                        # <!-- verbose
                        # for t in mdp.Q[mdp.start_state]:
                        #     print(mdp.Q[mdp.start_state][t])  
                        # verbose --!>
                        value = max([mdp.Q[mdp.start_state][t][i] for t in mdp.Q[mdp.start_state]])
                        # print(f'{item.description}: PR Value: {self.value_dict[item.description]}, New value: {value}')  # verbose
                        self.value_dict[item.description] = value
        # sum_values_abs = sum([self.value_dict[task.description] for task in tasks])
        # print(f'Root: {self.value_dict["Root"]}')
        # n = len(tasks)
        # bias = (-1 * sum_values_abs + 1)/n
        # print(f'Final Bias: {bias}')
        # value_list = []
        # for task in tasks:
        #     value_list.append(self.value_dict[task.description])
        # soft_values = softmax(value_list)
        # for i,task in enumerate(tasks):
        #     # print(f'{task.description}: Original Value: {self.value_dict[item.description]}, New value: {(self.value_dict[task.description] + bias) * sum_values_abs}')
        #     # self.value_dict[task.description] = (self.value_dict[task.description] + bias) * self.value_dict["Root"]
        #     print(f'{task.description}: Original Value: {self.value_dict[item.description]}, New value: {soft_values[i]* self.value_dict["Root"]}')
        #     self.value_dict[task.description] = soft_values[i] * self.value_dict["Root"]

        # for key in mdps:
        #     print(f'Key: {key}')
        # prs = self.task_pseudo_reward(mdps)
        #
        # pr_val = np.array([-1 * prs['id2pr'][task.description] for task in self.tasks])
        # inds = pr_val.argsort()
        # optimal_tasks = tasks[inds]
        # # print("Optimal Rewards")
        # sum_pr = 0
        # for task in optimal_tasks:
        #     sum_pr += prs['id2pr'][task.description]
        #     rr = np.round(prs['expected'][task.description], 3)
        #     pr = np.round(prs['id2pr'][task.description], 3)
        # #     print(f'Task: {task.description}, PRS: {pr}')
        # # print(f'Net PR Sum: {sum_pr}')
        return  self.value_dict# prs

    def solve_MDP(self, parent, children_nodes, start_time, verbose,flag=False):
        if len(children_nodes) == 0:  # No children, reached leaf node
            return
        # print(f'Solving treating goal as: {children_n¿odes[0].parent_item.description}')  # verbose
        if parent.is_completed():
        #     print(f'MDP for {parent.description} Need not solving. Already completed')  # verbose
            return
        tasks = deepcopy(children_nodes)
        root_val = sum([task.value for task in tasks])
        for task in tasks:
            if children_nodes[0].parent_item.description == "Root":  # If root node,
                # task.intrinsic_reward = task.value
                task.importance = task.value / root_val
                task.essential = False
        if parent.description == "Root":
            self.value_dict[parent.description] = root_val

            # task.value = self.value_dict[task.description]
        # print(f'{parent.description} Val: {self.value_dict[parent.description]}')  # verbose
        #             print(f'{task.description}, {task.value}, {task.intrinsic_reward}')

        mdp = MDP(items=tasks, value=self.value_dict[parent.description], end_time=self.end_time, gamma=self.gamma, loss_rate=self.loss_rate,
                 num_bins=self.num_bins, penalty_rate=self.penalty_rate, planning_fallacy_const=self.planning_fallacy_const,
                 slack_reward_rate=self.slack_reward_rate, start_time=start_time)
        mdp.solve(start_time=start_time, verbose=verbose)
        # print(f'Policy:')  # verbose
        s = mdp.start_state
        t_old = mdp.start_time
        slack_reward = mdp.get_slack_reward()
        policy_tasks = []
        while s[len(s)-1] != 1:
            pol = mdp.get_optimal_policy(s)
            a = next(iter(pol.values()))
            t = next(iter(pol.keys()))
            q_val = mdp.Q[s][t][a]
            # print(f'State: {s}, action: {a} Q_val: {q_val}')  # verbose
            # if q_val <= slack_reward:
            #     print(f'Slack Action Chosen. Action: {a}, q: {q_val} <= {slack_reward}')
            #     break
            # print(f'Action: {a}')
            if a is None or a is 'slack': 
                # policy_tasks.append('slack')
                a = len(s) - 1
            else:
                policy_tasks.append(mdp.items[a])
            s_ = mdp.exec_action(s, a)
            t_ = next(iter(pol.keys()))
            # print(f's, t: {s, t_old}, a: {a}, s_, t_: {s_, t}')
            # print(f'Options: {mdp.get_policy(s_)}')
            s = s_
            t_old = t

        # print(f'Q value')  # verbose
        best_q = np.NINF
        # print(f'Slack Reward of MDP: {slack_reward}')  # verbose
        # print(f'mdp.Q_s0: {mdp.Q_s0}')  # verbose
        # <!-- verbose
        # for keys in mdp.Q_s0:
        #     try:
        #         print(f'{keys.description}')
        #     except:
        #         print(f'{keys}')
        # print(f'mdp.Q_s0.values(): {mdp.Q_s0.values()}')
        # verbose -->)
        q = max(mdp.Q_s0.values())
        if q > best_q:
            best_q = q
        if best_q <= slack_reward:
            best_q = slack_reward
        # print(mdp.Q)
        # values = []
        values_abs = []
        left_tasks = []
        for task in mdp.items:
            if not task.is_completed():
                left_tasks.append(task)
        # print(f'Left Tasks: {left_tasks}')
        # print(f'best_q: {best_q}')  # verbose
        for i, task in enumerate(left_tasks):
            # print(f'Computing value for: {task.description}')  # verbose
            t_ = task.get_time_est()
            s_ = MDP.exec_action(mdp.start_state, i)

            q_ = max([max(mdp.Q[s_][t].values())for t in mdp.Q[s_]])
            tt = min(mdp.end_time, t_)
            cum_discount = mdp.get_cum_discount(tt)
            r = MDP.compute_total_loss(
                cum_discount=cum_discount, loss_rate=self.loss_rate)
            # print(f"R: {r}")
            if flag:
                # print(f' q_: {q_} Discount: {mdp.get_discount(tt)} Imm Reward: {task.get_immediate_reward()}')
            #     # print(f'{task.description} OLD: {(mdp.get_discount(tt) * q_ - best_q) } Immediate r: {task.get_immediate_reward()}')
                value = (mdp.get_discount(tt) * q_ - best_q) + task.get_immediate_reward() # -r
            #     # print(f'{task.description} gamma* V*(s_) - V*(s) + im: {(mdp.get_discount(tt) * q_ - best_q) + task.get_immediate_reward()}')
            #     # print(f'V*(s_) - V*(s): {q_ - best_q}, IM: {task.get_immediate_reward()}')
            #     # print(f'Gamma: {mdp.get_discount(tt)}, tt: {tt}')
            #     # print(f'gamma * V*(s_) - V*(s): {(mdp.get_discount(tt) * q_ - best_q)}, IM: {task.get_immediate_reward()}')
            else:
                value = (mdp.get_discount(mdp.end_time) * q_ - best_q) - r
            # value = max([mdp.Q[mdp.start_state][t][i] for t in mdp.Q[mdp.start_state]])
            # print(f'Discount: {mdp.get_discount(mdp.end_time)}, q_: {q_}, q: {q}')
            # print(f'{task.description}, reward: {task.get_expected_reward()} IM: {task.get_immediate_reward()} Q: {mdp.Q_s0[task]}')
            # print(f'Value: {value}')  # verbose
            # print(f'exp: {np.exp(value)}')
            # print(task.expected_rewards)
            # values.append(value)
            values_abs.append(value)
        # sum_values_abs = sum(values_abs)
        # print(f'Sum_values_abs: {sum_values_abs}')
        # n = len(mdp.items)
        # print(f'N: {n}')
        # bias = (-1 * sum_values_abs + 1)/n
        # print(f'Bias: {bias}')
        # print(f'Value_abs: {values_abs}')
        # values_abs = [j + bias for j in values_abs]
        # print(f'Value_abs: {values_abs}')
        # print(f' MDP Value: {mdp.value}')
        # print(f' SUM: {sum([values[i] + bias for i,_ in enumerate(mdp.items)])}')
        # print(f' SCALED SUM: {sum([(values[i] + bias) * mdp.value for i,_ in enumerate(mdp.items)])}')
        # max_val = max(values_abs)
        # if len(values_abs) <= 1:
        #     min_val = 0
        # else:
        #     min_val = min(values_abs)
        # scale = abs(max_val - min_val)
        # print(f'Max:{max_val}, Min: {min_val}')
        # print(f'Vals: {values_abs}')
        # print(f'Scale: {scale}')
        soft_values = softmax(values_abs)
        
        if len(policy_tasks) > 0:
            # for item in policy_tasks:
            #     print(f'{item.description}: Val: {item.value} Int: {item.get_intrinsic_reward()} Dict: {self.value_dict[item.description]}')
            sum_imp =  sum([item.importance for item in policy_tasks]) / sum([item.importance for item in mdp.items])
            # print(f'Sum_imp: {sum_imp}')  # verbose
            value = sum([item.get_intrinsic_reward() for item in policy_tasks]) + mdp.value * sum_imp
        else:
            value = 0
        intrinsic_sum = sum([task.get_intrinsic_reward() for task in mdp.items])
        # print(f'Scaling V: {value}')  # verbose
        for i, task in enumerate(left_tasks):
            self.value_dict[task.description] = soft_values[i] * value
            # self.value_dict[task.description] = (values_abs[i] + bias) * mdp.value
        # print(f'MDP Val: {mdp.value}. Parent: {self.value_dict[parent.description]}')
        # scaled_values = []
        # item_list = []
        # for i,task in enumerate(mdp.items):
        #     scaled_values.append(values[i] * (mdp.value / sum_values))
        #     item_list.append(task.description)
        # sorted_items = [x for _,x in sorted(zip(values, item_list))]
        # # for y,x in sorted(zip(values, scaled_values)):
        # #     print(f'Sorted Val: {y}, Scale: {x}')
        # print(f'Values: {values}')
        # print(f'Unsorted scaled values: {scaled_values}')
        # scaled_values.sort()
        # print(f'Sorted Scaled Values: {scaled_values}')
        # print(f'Sorted Items: {sorted_items}')
        # for i, task_desc in enumerate(mdp.items):
        #     self.value_dict[task_desc.description] = scaled_values[i]
        
        # <!-- verbose
        # print('Transferrred Q_s0 value down')
        # for task in mdp.Q_s0:
        #     if task != 'slack':
        #         print(f'{task.description} value dict:{self.value_dict[task.description]}')
        # verbose --!>
        return mdp
        # ================================= Computing Psuedo-rewards =================================
        # prs = mdp.compute_start_state_pseudo_rewards(flag=flag)
        # incentivized_tasks = prs["incentivized_items"]
        # for task in incentivized_tasks:
        #     self.pr_dict[task.description] = prs["id2pr"][task.description]
        #     self.optimal_dict[task.description] = prs["optimal"][task.description]
        #     self.expected_dict[task.description] = prs["expected"][task.description]
        # # Sort task in decreasing order w.r.t. optimal reward
        # pr_val =  [-1 * self.pr_dict[task.description] for task in incentivized_tasks]
        # # optimal_tasks = [x for _,x in sorted(zip(pr_val, incentivized_tasks))]
        # incentivized_tasks = np.array(incentivized_tasks)
        # pr_val = np.array(pr_val)
        # inds = pr_val.argsort()
        # optimal_tasks = incentivized_tasks[inds]
        # print("Optimal Rewards")
        # for task in optimal_tasks:
        #     print(task.description, self.pr_dict[task.description])
        #     self.value_dict[task.description] = self.pr_dict[task.description]
        # return

    def task_pseudo_reward(self, mini_mdps, scale=None, bias=None):
        start_time = self.start_time
        slack_reward = self.slack_reward
        # print(f'slack rewards: {slack_reward}')

        exp_reward = dict()
        scales = dict()
        biases = dict()
        sc_sum_prs = dict()
        next_q = dict()  # {Task Description: Q-value after item execution in s[0]}
        # Get V*(s)
        # Initialize best Q-value

        # Initialize {Task Description: pseudo-reward} dictionary
        id2pr = dict()

        for state in mini_mdps:
            best_q = np.NINF
            total_reward = 0
            # For each mini MDP, find pseudo-reward separately
            mdp = mini_mdps[state]
            q = max(mdp.Q_s0.values())
            # print(state)
            # for tt in mdp.Q_s0:
            #     print(tt.description, mdp.Q_s0[tt])
            # print(mdp.Q_s0)
            #         print(f'{state}: V* = {q} \nQ in mini-mdp: {mdp.Q_s0.values()}')
            if q > best_q:
                best_q = q

            if best_q <= slack_reward:
                best_q = slack_reward

            # To compute next Q
            for i, task in enumerate(mdp.items):
                # Suppose task is being done, what is the max Q value assuming task is going to be done
                #             print(task.description)
                exp_reward[task.description] = task.get_expected_reward()
                total_reward += task.get_expected_reward()
                t_ = task.get_time_est()
                s_ = MDP.exec_action(mdp.start_state, i)
                q_ = max(mdp.Q[s_][mdp.start_time + t_].values())
                next_q[task.description] = q_
            # print(f'best_q: {best_q}')

            # Initialize minimum pseudo-reward value (bias for linear transformation)
            min_pr = 0

            # Initialize sum of pseudo-rewards
            sum_pr = 0

            # Compute untransformed pseudo-rewards
            for task in mdp.items:
                pr = next_q[task.description] - best_q

                if np.isclose(pr, 0, atol=1e-6):
                    pr = 0

                # print(f'Task: {task.description} pr: {pr}')
                task.set_optimal_reward(pr)
                # Update minimum pseudo-reward
                min_pr = min(min_pr, pr)

                # Update sum of pseudo-rewards
                sum_pr += pr
            # print(f'sum_pr: {sum_pr}')

            sum_values = mini_mdps[state].value
            # print(f'Sum_Values: {sum_values}')

            if scale is None:
                # As defined in the report
                scale = 1.10

            # Set value of bias parameter
            if bias is None:
                # Total number of incentivized items
                n = len(mdp.items)
                # Derive value of the bias term
                bias = (sum_values - scale * sum_pr) / n

                # Take total reward into account
                bias -= (total_reward / n)
            # print(f'Bias: {bias}')

            # Sanity check for the sum of pseudo-rewards
            sc_sum_pr = 0

            # Perform linear transformation on item pseudo-rewards
            for task in mdp.items:
                # Transform pseudo-reward
                pr = f = scale * task.get_optimal_reward() + bias
                # pr = 0  # To see policy without pseudo reward
                # Get expected item reward
                task_reward = exp_reward[task.description]
                # print(f'{task.description}: r: {task_reward}, Next_Q: {next_q[task.description]}, f*: {task.get_optimal_reward()} , Final: {pr + task_reward}')
                # Add immediate reward to the pseudo-reward
                pr += task_reward
                # Store new (zero) pseudo-reward
                task.set_optimal_reward(pr)
                # Store pseudo-reward {Task Description: pseudo-reward}
                id2pr[task.description] = pr
                # If item is not slack action
                if task.get_idx() != -1:
                    # Update sanity check for the sum of pseudo-rewards
                    sc_sum_pr += task.get_optimal_reward()

            # assert np.isclose(sc_sum_pr, sum_values, atol=1e-3)
            sc_sum_prs[state] = sc_sum_pr
            scales[state] = scale
            biases[state] = bias
            # print(f"\nTotal sum of pseudo-rewards: {sc_sum_pr:.2f}\n")
        return {
            "expected": exp_reward,
            "id2pr": id2pr,
            "sc_sum_pr": sc_sum_prs,
            "scale": scales,
            "bias": biases
        }

    def scale_bias(self, bias=None, scale=None):
        '''

        :param bias:
        :param scale:
        :return:
        '''
        # print(f'****** In Scale Bias ******')
        sum_pr = sum([self.optimal_dict[task.description] for task in self.tasks])
        sum_item_values = sum([goal.value for goal in self.tree["Root"]]) + sum([task.intrinsic_reward for task in self.tasks])
        total_reward = sum([self.expected_dict[task.description] for task in self.tasks])
        # print(f'Sum Of all Values: {sum_item_values}')
        # Set value of scaling parameter
        if scale is None:
            # As defined in the report
            scale = 1.1

        # Set value of bias parameter
        if bias is None:
            # Total number of incentivized items
            n = len(self.tasks)

            # Derive value of the bias term
            bias = (sum_item_values - scale * sum_pr) / n

            # Take total reward into account
            bias -= (total_reward / n)
        # print(f'Bias: {bias}')
        # Initialize {item ID: pseudo-reward} dictionary
        id2pr = dict()

        # Perform linear transformation on item pseudo-rewards
        for item in self.tasks:
            # Get item unique identification
            item_id = item.description

            # Transform pseudo-reward
            pr = f = scale * self.optimal_dict[item.description] + bias
            # Get expected item reward
            item_reward = self.expected_dict[item.description]

            # print(f'{item.description}: r: {item_reward}, pr: {pr}, new_pr: {pr + item_reward}')
            # print(f'Scale: {scale} Pre Scale: {self.optimal_dict[item.description]}, Bias: {bias} Final: {pr + item_reward}')
            # Add immediate reward to the pseudo-reward
            pr += item_reward

            # Store pseudo-reward {item ID: pseudo-reward}
            id2pr[item_id] = pr

        return {
            "id2pr": id2pr,
            "scale": scale,
            "bias": bias
        }

    def get_actions(self, task_name_list):
        action_list = []
        for i, task in enumerate(self.tasks):
            if task.description in task_name_list:
                action_list.append(i)
        return action_list

    def solve_exact(self, start_time=None):
        nA = self.num_tasks + 1

        def one_step_lookahead(curr_state):
            # print(f'Looking ahead for state: {curr_state}')
            s = curr_state["s"]
            t = curr_state["t"]

            # Initialize policy entries for state and (state, time))
            self.Pexact.setdefault(s, dict())

            # Initialize Q-function entry for the (state, time) pair
            self.Qexact.setdefault(s, dict())
            self.Qexact[s].setdefault(t, dict())

            # Initialize reward entries for state and (state, time))
            self.Rexact.setdefault(s, dict())
            self.Rexact[s].setdefault(t, dict())
            if s[self.num_tasks] != 1:
                for action in range(nA):
                    if s[action] == 0: # task is not done
                        # print(f'Evaluating for action: {action}')
                        # check if with action, is its parent being completed. i.e. for a goal, all
                        s_ = list(s)
                        s_[action] = 1
                        s_ = tuple(s_)
                        # print(f's: {s}, S_: {s_}')
                        # Probability of transition to next state
                        prob = 1

                        # The computation not already been done
                        if action not in self.Qexact[s][t].keys():
                            if s_[self.num_tasks] != 1:
                                task = self.tasks[action]  # task which action will complete
                                goals = tuple(self.goal_tasks.keys())
                                goal_name = goals[[task.description in self.goal_tasks[goal_des] for goal_des in self.goal_tasks].index(True)]
                                for goal in self.tree['Root']:
                                    if goal.description == goal_name:
                                        break
                                # print(f'{task.description} in {goal_name}')

                                # Initialize expected value for action
                                self.Qexact[s][t].setdefault(action, 0)

                                # Initialize entry for (state, time, action)
                                self.Rexact[s][t].setdefault(action, 0)

                                task_deadline = task.get_deadline()
                                time_transitions = task.get_time_transitions().items()

                                #  Check if new state is completed
                                flag = True
                                sum_imp_completed = 0
                                action_list = self.get_actions(self.goal_tasks[goal_name])
                                sum_imp = sum([self.tasks[a].importance for a in action_list])
                                for a in action_list:
                                    if s_[a] == 0:
                                        if self.tasks[a].essential: # there is an uncompleted essential task for goal
                                            flag = False
                                    else:  # only  used if essential
                                        sum_imp_completed += self.tasks[a].importance
                                # print(f'In state: {s_}, Flag: {flag}, sum_imp_comp: {sum_imp_completed}, sum_imp: {sum_imp}')
                                beta = 0
                                exp_time_reward = 0
                                exp_total_reward = 0
                                for time_est, prob_t_ in time_transitions:
                                    # Make time transition
                                    t_ = t + time_est

                                    # Initialize Q-values for state' and time'
                                    self.Qexact.setdefault(s_, dict())
                                    self.Qexact[s_].setdefault(t_, dict())

                                    # Get cumulative discount w.r.t. item duration
                                    # Cognitive/time cost
                                    cum_discount = self.get_cum_discount(time_est)
                                    r = MDP.compute_total_loss(
                                        cum_discount=cum_discount, loss_rate=self.loss_rate
                                    )

                                    # Add deadline to the missed deadlines if not attained
                                    if task.is_deadline_missed(t_):
                                        # print(f'Task: {task.description} Deadline Missed')
                                        # Compute total penalty for missing item deadline
                                        total_penalty = \
                                            self.penalty_rate * (t_ - task_deadline)

                                        # Update penalty
                                        beta += prob_t_ * total_penalty
                                        if flag:
                                            r += (goal.value* (sum_imp_completed/ sum_imp))/ (1+beta)
                                    r += task.intrinsic_reward
                                    exp_time_reward += prob_t_ * r

                                    # Generate next state
                                    state_dict = {
                                        "s": s_,
                                        "t": t_
                                    }

                                    # Explore the next state
                                    one_step_lookahead(state_dict)
                                    # Get best action and its respective for (state', time')
                                    a_, r_ = MDP.max_from_dict(self.Qexact[s_][t_])
                                    self.Pexact[s_][t_] = a_

                                    # - Multiple time-step discount (SMDP)
                                    total_reward = r + self.get_discount(time_est) * r_
                                    exp_total_reward += prob_t_ * total_reward

                                # Add more values to the expected value
                                self.Qexact[s][t][action] += prob * exp_total_reward
                                self.Rexact[s][t][action] += prob * exp_time_reward
            # ===== Terminal state ===== (All children visited)
            if sum(s[:-1]) == len(s[:-1]):
                # Initialize dictionary for the terminal state Q-value
                self.Qexact[s][t].setdefault(None, 0)

                # Compute reward for reaching terminal state s in time t
                self.Rexact[s][t].setdefault(None, 0)
            elif s[self.num_tasks] == 1:
                self.Qexact[s][t].setdefault(None, self.slack_reward)
                self.Rexact[s][t].setdefault(None, self.slack_reward)


        # Initialize start state & time
        t = self.start_time if start_time is None else start_time
        s =  tuple(0 for _ in range(self.num_tasks + 1))
        curr_state = {
            "s": s,
            "t": t
        }
        # Start iterating
        # print(f'self.gamma: {self.gamma}')
        one_step_lookahead(curr_state)

        policy = []
        # Get best action in the start state and its corresponding reward
        a, r = MDP.max_from_dict(self.Qexact[s][t])
        # Store policy for the next (state, time) pair
        self.Pexact[s][t] = a

        return  {
            "P": self.Pexact,
            "Q": self.Qexact
        }

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def num_to_binary(num):
    bin_str = bin(num);
    bin_seg = bin_str.split('b')[1]
    return tuple(bin_seg)

def binary_to_num(binary):
    return int('0b'+binary, 2)

