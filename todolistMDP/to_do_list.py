import numpy as np
import time
from toolz import memoize, compose
from collections import deque
from copy import deepcopy
from todolistMDP.zero_trunc_poisson import get_binned_dist
from pprint import pprint
import itertools

class Item:
    """
    All tasks, sub-goals are effectively a Item. An Item can have a single parent and can itself be a parent to multiple Items.
    """
    def __init__(self, description, completed=False, deadline=None,
                 deadline_datetime=None, item_id=None, children=None,
                 parent_item=None, essential=False, importance=0, time_est=0, today=None, value=None,
                 intrinsic_reward=0, end_time=np.PINF, gamma=1.0, loss_rate=0.,
                 num_bins=1, penalty_rate=0., planning_fallacy_const=1.,
                 slack_reward_rate=0, start_time=0):
        """

        :param description: name of Item
        :param completed: Flag if Item is completed. In case of sub-Items: An item is said to be complete if all its
                          essential super-ordinate items are completed.
        :param deadline: deadline of Item
        :param deadline_datetime: deadline of Item in datetime structure
        :param item_id: Item ID
        :param children: Immediate sub items to be considered
        :param parent_item: Parent of current item
        :param essential: Flag if item is set to be essential to complete super-ordinate item.
                          All goal items are automatically considered essential
        :param importance: Degree of importance to completion of super-ordinate item. In scale of 1.
        :param time_est: Time estimate to complete task
        :param today: Flag if item is marked to be completed today
        :param value: Value for completing of super-ordinate item
        :param intrinsic_reward: Intrinsic reward for completing item

        :param end_time: End time of the MDP (i.e. horizon)
        :param gamma: Discount factor
        :param slack_reward_rate: Unit-time reward for slack-off action.
        :param start_time:  Starting time of the MDP
        """

        # Compulsory parameters
        self.description = description

        self.end_time = end_time
        self.gamma = gamma
        self.loss_rate = loss_rate
        self.num_bins = num_bins
        self.penalty_rate = penalty_rate
        self.planning_fallacy_const = planning_fallacy_const
        self.slack_reward_rate = slack_reward_rate
        self.start_time = start_time

        # Optional parameters
        self.completed = completed
        self.deadline = deadline
        self.deadline_datetime = deadline_datetime
        self.parent_item = parent_item
        self.time_est = time_est
        self.today = today
        self.value = value
        self.intrinsic_reward = intrinsic_reward
        self.essential = essential
        self.importance = importance

        self.item_id = item_id
        if self.item_id is None:
            self.item_id = self.description

        # Initialize expected reward
        self.expected_rewards = []

        # Initialize optimal reward
        self.optimal_reward = np.NINF

        # Initialize dictionary of Q-values at (s, t): (0, 0)
        self.Q_s0 = dict()

        # Initialize start time
        self.start_time = None

        # Initialize queue of tasks
        self.tasks = None

        # Initialize time transitions
        self.time_transitions = {
            self.time_est: 1.
        }

        # Initialize today items set //for children nodes
        self.today_children = set()

        # Initialize dictionary of maximum future Q-values
        self.future_q = {
            None: None
        }

        # Initialize index
        self.idx = None

        # Initialize computations
        self.already_computed_pruning = 0
        self.total_computations = 0

        # Initialize list of sub-items
        self.children = deque()

        # Initialize complete flag of all essential sub-items
        self.essential_items_complete = deque()

        # Initialize importance of all sub-items
        self.importance_items = deque()

        # Index of all essential sub-items
        self.essential_child = deque()

        # Initialize sibllings to execute
        self.siblings = None

        # Initialize number of sub-items
        self.num_children = 0

        # Initialize 0-th state
        self.start_state = tuple()

        # Calculate total time estimate of the to-do list from Item
        self.total_time_est = 0

        # "Cut" horizon in order to reduce the number of computations
        self.end_time = min(self.end_time, self.total_time_est)

        # Add items on the next level/depth
        if children is not None:
            self.add_children(children)
            self.time_est = sum([child.time_est for child in self.children])
        else:
            self.children = []
            self.num_children = 0

        # Initialize policy, Q-value function and pseudo-rewards
        self.P = dict()  # Optimal policy {state: action}
        self.Q = dict()  # Action-value function {state: {action: value}}
        self.R = dict()  # Expected rewards

    def __hash__(self):
        return id(self)

    def __str__(self):
        return f"{self.description} " \
               f"~~{self.time_est} " \
               f"=={self.value}, " \
               f"Intrinsic Value:{self.intrinsic_reward}, " \
               f"DUE:{self.deadline}, " \
               f"Importance: {self.importance}, "\
               f"Essential: {self.essential}, "\
               f"Completed: {self.completed}, "\
               # f"ID: {hex(id(self))}" \

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

    def add_children(self, children, available_time=np.PINF, prepare_solve=False):
        """

        :param children:
        :param available_time:
        :param prepare_solve:
        :return:
        """

        # Initialize state for the next tasks
        appended_state = list(0 for _ in range(len(children)))

        if prepare_solve:
            self.children = deque()
            self.num_children = 0
            self.today_children = set()
            self.start_state = appended_state
        else:
            self.start_state = list(self.start_state) + appended_state

        self.importance = 0
        self.intrinsic_reward = 0
        for idx, child in enumerate(children):
            # Shift children index
            idx += self.num_children

            # If item has no deadline, set super-item's deadline
            if not prepare_solve:
                child.set_deadline(self.deadline, compare=True)

            # Set item index
            child.set_idx(idx)
            self.importance += child.importance
            self.intrinsic_reward += child.intrinsic_reward
            self.compute_binning(self.num_bins)

            # Add child that has to be executed today
            # print(f'child: {child}, is_today: {child.is_today()}, {child.is_completed()}')
            if child.is_today() and not child.is_completed():
                self.today_children.add(child)

            # Set child as completed in the start state
            if child.is_completed():
                self.start_state[idx] = 1
                self.done_children.add(child)

            # Add child
            self.children.append(child)

            # Set goal as a parent item
            if prepare_solve:
                child.set_parent_item(self)

        # Convert start state from list to tuple
        self.start_state = tuple(self.start_state)
        # print(f'{self.description} start_state: {self.start_state}')
        # Set number of children_
        self.num_children = len(self.start_state)

        # if prepare_solve:

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

    def get_best_action(self, slack_reward):

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

    def get_expected_reward(self):
        return np.mean(self.expected_rewards)

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

    def get_children(self):
        return self.children

    def get_children_to_iterate(self):
        return self.children_to_iterate

    def get_num_children(self):
        return self.num_children

    def get_optimal_reward(self):
        return self.optimal_reward

    def get_parent_item(self):
        return self.parent_item

    def done_children(self, curr_state):
        done_children = []
        for idx in range(len(curr_state["s"])):
            if self.children[idx]:
                done_children.append(self.children[idx])
        return tuple(done_children)

    def get_reward(self, curr_state=None, beta=0., discount=1.):
        if curr_state is None:
            curr_state = self.start_state
        # print(f'get_reward --> {self.description}')
        # print(f'intrinsic_reward {self.intrinsic_reward}')
        reward = discount * (self.get_intrinsic_reward() + (self.extrinsic_reward(curr_state) / (1 + beta)))
        # print(f'Reward: {reward}')
        return reward

    def extrinsic_reward(self, curr_state):
        # print(f'In Extrinsic Reward: curr_state["s"]: {curr_state["s"]}')
        if self.parent_item is None:  # Goal node
            parent_item = self
        else:
            parent_item = self.parent_item
        # print(f'Parent Item:  {parent_item.description}')
        state_complete = True
        # print(f'{self.description} LEN: {len(curr_state["s"])}')
        for idx in range(len(curr_state["s"])):
            # print(f'idx: {idx}')
            # print(len(parent_item.children))
            if not (parent_item.children[idx].essential and curr_state["s"][
                idx]):  # Basically if any essential item is not complete, set flag to False
                state_complete = False
                break
        if not state_complete:
            # print('Is not Completed')
            return 0
        else:
            # print(f'{self.value}')
            # print(f'{[child.get_importance() for child in parent_item.done_children(curr_state)]}')
            # print(f'{parent_item.get_importance()}')
            return self.value * sum(
                    [child.get_importance() for child in parent_item.done_children(curr_state)]) / parent_item.get_importance()

    def get_essential(self, action):
        return self.children[action].is_essential()

    def get_intrinsic_reward(self):
            return self.intrinsic_reward

    def get_task_list(self):
        return self.tasks

    def get_time_est(self):
        return self.time_est

    def get_time_transitions(self, t=None):
        if t is None:
            return self.time_transitions
        return self.time_transitions[t]

    def get_value(self):
        return self.value

    def get_importance(self):
        return self.importance

    def get_intrinsic(self):
        return self.intrinsic_reward

    def init_task_list(self):
        self.tasks = deque()

    def is_completed(self):
        return self.completed

    def check_completed(self):
        if self.completed:
            return self.completed
        elif self.children is None:
            return self.is_completed()
        elif len(self.children) == 0:  # No sub-items, own tag sets completeness
            return self.is_completed()
        else:  # if sub-items, all essential sub-items need to be completed
            for child in self.children:
                # print(f'{item.description} Essential-> {item.essential} ID: {hex(id(item))}')
                if child.essential:
                    # print(f'{item.description}: {item.is_completed()}')
                    if not child.check_completed():
                        return self.completed  # any essential sub-item not completed, return False
            self.set_completed()  # if all complete, set tag to complete
            # print(f'Setting Completed {self.is_completed()}')
        return self.is_completed()

    def is_deadline_missed(self, t):
        return t > self.deadline

    def is_today(self):
        return self.today

    def is_essential(self):
        return self.essential

    def parse_sub_goals(self, min_time=0, max_time=0):
        # TODO: Add comments (!)

        if max_time == float('inf'):
            return self.children

        # If no limit has been defined
        if max_time == 0:
            return self.tasks

        sub_goals = deque()

        # If the time estimate fits the time limit
        if min_time <= self.time_est <= max_time:

            # If it is a goal node
            if self.parent_item is None:
                sub_goals.extend(self.children)

            # If it is an item within a goal
            else:
                sub_goals.append(self)

        # If it is a leaf node
        elif self.num_children == 0:

            sub_goals.append(self)

        else:

            for child in self.children:
                child_sub_goals = child.parse_sub_goals(
                    min_time=min_time, max_time=max_time)
                sub_goals.extend(child_sub_goals)

        return sub_goals

    def print_recursively(self, level=0, indent=2):
        print(f"{' ' * (level * indent)}- {self}")
        for child in self.children:
            child.print_recursively(level + 1, indent=indent)

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
        # print(f'In set_expected_reward {self.description}')
        self.expected_rewards.append(expected_reward)

    def set_future_q(self, t, value, compare=False):

        # Set future Q-value
        self.future_q.setdefault(t, value)

        if compare:
            self.future_q[t] = max(self.future_q[t], value)

    def set_idx(self, idx):
        self.idx = idx

    def set_essential(self, essential):
        self.essential = essential

    def set_importance(self, importance):
        self.importance = importance

    def set_children_to_iterate(self, children_to_iterate):
        self.children_to_iterate = children_to_iterate

    def set_optimal_reward(self, optimal_reward):
        self.optimal_reward = optimal_reward

    def set_parent_item(self, parent_item):
        self.parent_item = parent_item

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_time_est(self, time_est):
        self.time_est = time_est

    def set_today(self, today):
        self.today = today

    def set_value(self, value):
        self.value = value

    # ================================ SOLVERS =================================
    def solve(self, params, start_time=None, verbose=False):

        def solve_next_child(curr_state, verbose=False):
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
            # Initialize next child
            next_child = None

            # Initialize policy entries for state and (state, time))
            self.P.setdefault(s, dict())

            # Initialize Q-function entry for the (state, time) pair
            self.Q.setdefault(s, dict())
            self.Q[s].setdefault(t, dict())

            # Initialize reward entries for state and (state, time))
            self.R.setdefault(s, dict())
            self.R[s].setdefault(t, dict())
            # Find the next uncompleted goal

            for child_idx in range(self.num_children):
                # If the goal with index goal_idx is not completed
                if s[child_idx] == 0:
                    # Increase total-computations counter
                    self.total_computations += 1

                    # Set action to be the corresponding goal index
                    a = child_idx
                    # print(f'In State {s}, Action considered: { a}')
                    # Generate next state
                    s_ = Item.exec_action(s, a)  # s_ is a different object to s

                    # Get next child object
                    next_child = self.children[child_idx]

                    # TODO: Probability of transition to next state
                    prob = 1

                    # The computation has already been done --> Prune!
                    if a in self.Q[s][t].keys():

                        # Increase already-computed-pruning counter
                        self.already_computed_pruning += 1

                        if verbose:
                            print(f"Transition (s, t, a, t') {(s, t, a, t_)} "
                                  f"already computed.")

                    # Explore the next subgoal-level state
                    else:
                        # Initialize expected value for action
                        self.Q[s][t].setdefault(a, 0)

                        # Initialize entry for (state, time, action)
                        self.R[s][t].setdefault(a, 0)
                        # Get deadline time for next item
                        task_deadline = next_child.get_deadline()

                        # Get time transitions of the next state
                        time_transitions = next_child.get_time_transitions().items()
                        exp_task_reward = 0
                        beta = 0
                        for time_est, prob_t_ in time_transitions:

                            # Increase total-computations counter
                            self.total_computations += 1

                            # Make time transition
                            t_ = t + time_est

                            # Initialize Q-values for state' and time'
                            self.Q.setdefault(s_, dict())
                            self.Q[s_].setdefault(t_, dict())

                            # Get cumulative discount w.r.t. item duration
                            cum_discount = ToDoList.get_cum_discount(time_est)

                            # Calculate total loss for next action (immediate "reward")
                            r = ToDoList.compute_total_loss(
                                cum_discount=cum_discount, loss_rate=params["loss_rate"]
                            )

                            # Add deadline to the missed deadlines if not attained
                            if next_child.is_deadline_missed(t_):
                                # Compute total penalty for missing item deadline
                                total_penalty = \
                                    params["penalty_rate"] * (t_ - task_deadline)

                                # Update penalty
                                beta += prob_t_ * total_penalty
                            #  DEBUG Mode
                            if next_child.is_deadline_missed(t_):
                                print(f'{self.description} state: {s} action: {a} Deadline Missed.')
                                r += 0
                                # r+= next_child.get_reward(curr_state={"s": s_}, beta=beta, discount=1)
                            else:
                                r += next_child.get_reward(curr_state={"s": s_}, beta=beta, discount=1)
                            # Update expected reward for (state, time, action)
                            exp_task_reward += prob_t_ * r
                        if s == tuple(0 for _ in range(self.num_children)):
                            # print(f'In solve_next_child: state: {s}')
                            # print(f'{next_child.description}: exp_reward: {exp_task_reward}')
                            next_child.set_expected_reward(exp_task_reward)


                        self.R[s][t][a] += prob * r

                        # Generate next goal-level state
                        state_dict = {
                            "s": s_,
                            "t": t_
                        }

                        # Explore the next goal-level state
                        solve_next_child(state_dict, verbose=verbose)

                        # Get best action and its respective for (state', time')
                        a_, r_ = Item.max_from_dict(self.Q[s_][t_])

                        # Store policy for the next (state, time) pair
                        self.P[s_][t_] = a_

                        # Store future Q-value
                        next_child.set_future_q(t_, r_, compare=True)

                        # Compute total reward for the current state-time action as
                        # immediate + (discounted) expected future reward
                        # - Single time-step discount (MDP)
                        # total_reward = r + self.gamma ** next_goal.num_tasks * r_

                        # - Multiple time-step discount (SMDP)
                        total_reward = r + ToDoList.get_discount(time_est) * r_

                        # TODO: Probability of transition to next state
                        prob = 1

                        # Add more values to the expected value
                        self.Q[s][t][a] += prob * total_reward

                # Store initial-state Q-value (if initial time applies)
                if t == self.start_time:
                    self.Q_s0[next_child] = self.Q[s][t][a]

            # ===== Terminal state ===== (All children visited)
            if next_child is None:
                # Initialize dictionary for the terminal state Q-value
                self.Q[s][t].setdefault(None, 0)

                # Compute reward for reaching terminal state s in time t
                self.R[s][t].setdefault(None, 0)

        # Initialize start state & time
        t = self.start_time if start_time is None else start_time
        s = tuple(0 for _ in range(self.num_children))


        curr_state = {
            "s": s,
            "t": t
        }

        # Start iterating
        solve_next_child(curr_state, verbose=verbose)

        # Get best action in the start state and its corresponding reward
        a, r = Item.max_from_dict(self.Q[s][t])

        # Store policy for the next (state, time) pair
        self.P[s][t] = a

        return r

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
        self.R = dict()  # Expected rewards

        # Initialize computation counters
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

    @staticmethod
    def compute_total_loss(cum_discount, loss_rate):
        return loss_rate * cum_discount

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

            if goal.check_completed():
                self.start_state[idx] = 1

        # Convert list to tuple
        self.start_state = tuple(self.start_state)

    def compute_slack_reward(self):
        if self.slack_reward_rate == 0:
            return 0

        if self.gamma < 1:
            return self.slack_reward_rate * (1 / (1 - self.gamma))

        return np.PINF

    def get_optimal_policy(self, state=None):
        """
        Returns the mapping of state to the optimal policy at that state
        """
        if state is not None:
            return self.P[state]
        return self.P

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

    def solve(self, in_depth=True, verbose=False):

        params = {
            "loss_rate": self.loss_rate,
            "penalty_rate": self.penalty_rate
        }

        def solve_next_goals(curr_state, in_depth=in_depth, verbose=False):
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
                    print(f'Goal state: {s}, next goal: {goal_idx}')
                    # Increase total-computations counter
                    self.total_computations += 1

                    # Set action to be the corresponding goal index
                    a = goal_idx

                    # Generate next state
                    s_ = ToDoList.exec_action(s, a)

                    # Get next Goal object
                    next_goal = self.goals[goal_idx]
                    # print(f'Goal: {next_goal.description}')

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
                                            start_time=t,
                                            verbose=verbose)

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
        solve_next_goals(curr_state, in_depth=in_depth, verbose=verbose)

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


class MainToDoList:
    # Class attributes
    DISCOUNTS = None
    CUM_DISCOUNTS = None

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

        for goal in self.complete_to_do_list:
            for item in goal.children:
                self.set_parent(item, goal)
        # Get tasks
        self.tasks = tuple(self.flatten(self.get_tasks(self.complete_to_do_list)))
        self.num_tasks = len(self.tasks)
        # Set nodes from the nested information
        self.nodes = tuple(self.flatten(self.get_nodes(self.complete_to_do_list, False)))
        self.node_names = tuple(self.flatten(self.get_nodes(self.complete_to_do_list, True)))

        # Set Value dictioonary used to store optimal rewards
        value_dict = dict()
        for node in tuple(self.node_names):
            value_dict[node] = 0
        self.value_dict = value_dict
        self.tree = dict()
        self.tree["0"] = self.complete_to_do_list
        self.tree_recurse(self.complete_to_do_list)

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

    def tree_recurse(self, to_do_list):
        for goal in to_do_list:
            self.tree[goal.description] = goal.children
            self.tree_recurse(goal.children)

    def get_nodes(self, complete_to_do_list, desc=False):
        nodes = []
        for goal in complete_to_do_list:
            if desc == True:
                nodes.append(goal.description)
            else:
                nodes.append(goal)
            for child in goal.children:
                nodes.append(self.get_nodes([child], desc))
        return nodes

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
    def recurse(goal):
        if len(goal.children) == 0:
            return goal
        rcg = [MainToDoList.recurse(child) for child in goal.children]
        return rcg

    def get_tasks(self, complete_to_do_list):
        all_tasks = []
        for goal in complete_to_do_list:

            rcg = MainToDoList.recurse(goal)
            rcg = list(itertools.chain(*rcg))
            # for rr in rcg:
            all_tasks.extend(rcg)
        return all_tasks

    def set_parent(self, node, parent):
        node.parent_item = parent
        if len(node.children) != 0:
            for c in node.children:
                self.set_parent(c, node)

    def mini_mdp(self, list_of_items, start_time=None):
        if start_time is None:
            start_time = 0
        copy_items = deepcopy(list_of_items)
        #         print(f'copy_items: {id(copy_items)}')
        #         print(f'list_of_items: {id(list_of_items)}')
        for goal in copy_items:
            for child in goal.children:
                child.children = None
        to_do_list = ToDoList(
            copy_items,
            gamma=self.gamma,
            loss_rate=self.loss_rate,
            num_bins=self.num_bins,
            penalty_rate=self.penalty_rate,
            slack_reward_rate=self.slack_reward_rate,
            start_time=start_time
        )
        return to_do_list

    def solve_mini_mdp(self, to_do_list, start_time=None, verbose=False):
        if start_time is None:
            start_time = self.start_time
        print(f'Will be solving with Items')
        node_names = tuple(self.flatten(self.get_nodes(to_do_list, True)))
        print(node_names)

        # Make MDP to now solve
        mini_MDP = self.mini_mdp(to_do_list, start_time)
        # ================================= Solving MDP =================================
        mini_MDP.solve(verbose=False)

        # ================================= Computing Psuedo-rewards =================================
        prs = self.compute_start_state_pseudo_rewards(
            mini_MDP)
        incentivized_tasks = prs["incentivized_items"]
        # Convert task queue to a list
        optimal_tasks = list(incentivized_tasks)
        # Sort task in decreasing order w.r.t. optimal reward
        optimal_tasks.sort(key=lambda task: -task.get_optimal_reward())
        for task in optimal_tasks:
            print(task.description, task.get_optimal_reward())
            self.value_dict[task.description] = task.get_optimal_reward()
        return

    def solve(self, to_do_list=None, start_time=None, verbose=False):
        print(f'*********** In MainToDoList solve ***********')
        if start_time is None:
            start_time = self.start_time
        if to_do_list is None:
            to_do_list = self.complete_to_do_list
        params = {
            "loss_rate": self.loss_rate,
            "penalty_rate": self.penalty_rate
        }

        def recurse_solve(to_do):
            print(f'### In recurse solve ###')
            # First solve the goal and sub-goals as terminal Items
            self.solve_mini_mdp(to_do, start_time, verbose=verbose)
            # Then update value of sub_goals
            for goal in to_do:
                flag = 0
                for item in goal.children:
                    item.value = self.value_dict[item.description]
                    flag += item.num_children
                if flag:  # If sub goals are leaf nodes, no longer need to recurse downwards
                    # Then solve for EACH goal
                    recurse_solve(goal.children)

        recurse_solve(to_do_list)
        return self.value_dict

    def compute_start_state_pseudo_rewards(self, to_do_list: ToDoList,
                                           bias=None, scale=None):
        print(f'****** In compute_start_state_pseudo_rewards ******')
        # Get list of goals
        goals = to_do_list.get_goals()

        # Get goal future Q-values
        item_q = dict()  # {item ID: Q-value for item execution in s[0]}
        next_q = dict()  # {item ID: Q-value after item execution in s[0]}

        # Initialize best Q-value
        best_q = np.NINF

        # Initialize start time
        start_time = to_do_list.get_start_time()
        # print("Start Time: {}".format(start_time))

        # Initialize list of incentivized items
        incentivized_items = deque()

        # Initialize total reward
        total_reward = 0

        # Initialize slack reward
        slack_reward = to_do_list.get_slack_reward()
        # print(f'slack rewards: {slack_reward}')
        # print(goals)
        for goal in goals:

            if best_q <= slack_reward:
                best_q = slack_reward

            # Get time step after executing goal
            t_ = goal.get_time_est()

            # Set future Q-value
            future_q = goal.get_future_q(start_time + t_)
            # print(f'future q: {future_q}')
            # print(f'goal.Q_s0.items(): {goal.Q_s0.items()}')
            # Compute Q-value in next step
            #             print(f'goal.Q_s0: {goal.Q_s0}')
            for item, q in goal.Q_s0.items():
                # print(f'Item: {item.description}')
                # print(item.expected_rewards)
                # Update Q-value with Q-values of future goals
                q += future_q

                # Get item ID
                item_id = item.get_id()
                # print(f'item_id: {item_id}')
                # Store Q-value for item execution in s[0]
                item_q[item_id] = q
                # print(f'q: {q}')
                # Get expected item reward
                reward = item.get_expected_reward()

                # Update total reward
                #                 print(f'r: {reward}')
                #                 print(f'{total_reward}')

                total_reward += reward

                # Compute Q-value after transition
                next_q[item_id] = q - reward
                # print(f'Next Q: {q - reward}')

                # Update best Q-value and best next Q-value
                if best_q <= q:
                    best_q = q

                # Add items to the list of incentivized items (?!)
                incentivized_items.append(item)

        # Initialize minimum pseudo-reward value (bias for linear transformation)
        min_pr = 0

        # Initialize sum of pseudo-rewards
        sum_pr = 0
        # print(f'Init sum_pr: {sum_pr}')
        # Compute untransformed pseudo-rewards
        for item in incentivized_items:

            # Get item ID
            item_id = item.get_id()

            # Compute pseudo-reward
            pr = next_q[item_id] - best_q

            if np.isclose(pr, 0, atol=1e-6):
                pr = 0

            # Store pseudo-reward
            # print(f'{item.description}: optimal reward:{pr}')
            item.set_optimal_reward(pr)

            # Update minimum pseudo-reward
            min_pr = min(min_pr, pr)

            # Update sum of pseudo-rewards
            sum_pr += pr
            # print(f'sum_pr: {sum_pr}')

        # Compute sum of goal values
        sum_goal_values = sum([goal.value + goal.intrinsic_reward for goal in goals])
        print(f'Sum Of all Values: {sum_goal_values}')
        # Set value of scaling parameter
        if scale is None:
            # As defined in the report
            scale = 1.10

        # Set value of bias parameter
        if bias is None:
            # Total number of incentivized items
            n = len(incentivized_items)

            # Derive value of the bias term
            bias = (sum_goal_values - scale * sum_pr) / n

            # Take total reward into account
            bias -= (total_reward / n)

        # print("Bias:", bias)
        # print("Scale:", scale)
        # print()

        # Initialize {item ID: pseudo-reward} dictionary
        id2pr = dict()

        # Sanity check for the sum of pseudo-rewards
        sc_sum_pr = 0
        # print(f'Init sc_sum_pr: {sc_sum_pr}')

        # Perform linear transformation on item pseudo-rewards
        for item in incentivized_items:

            # Get item unique identification
            item_id = item.get_id()

            # Transform pseudo-reward
            pr = f = scale * item.get_optimal_reward() + bias
            # print
            # Get expected item reward
            item_reward = item.get_expected_reward()

            print(f'{item.description}: r: {item_reward}, pr: {pr}, new_pr: {pr + item_reward}')
            print(f'Scale: {scale} Optimal Reward: {item.get_optimal_reward()}, Bias: {bias}')
            # Add immediate reward to the pseudo-reward
            pr += item_reward

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

            # Store new (zero) pseudo-reward
            item.set_optimal_reward(pr)

            # If item is not slack action
            if item.get_idx() != -1:
                # Update sanity check for the sum of pseudo-rewards
                sc_sum_pr += item.get_optimal_reward()

            # Store pseudo-reward {item ID: pseudo-reward}
            id2pr[item_id] = pr

        print(f"\nTotal sum of pseudo-rewards: {sc_sum_pr:.2f}\n")

        return {
            "incentivized_items": incentivized_items,

            "id2pr": id2pr,
            "sc_sum_pr": sc_sum_pr,
            "scale": scale,
            "bias": bias
        }

