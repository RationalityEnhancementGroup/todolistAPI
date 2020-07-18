import numpy as np
import time

from collections import deque
from copy import deepcopy
from math import ceil
from todolistMDP.zero_trunc_poisson import get_binned_distrib
from pprint import pprint


class Item:
    def __init__(self, description, completed=False, deadline_datetime=None,
                 idx=None, item_id=None, loss_rate=None, num_bins=None,
                 planning_fallacy_const=None, rewards=None, time_est=None,
                 unit_penalty=None):
        self.description = description
        self.completed = completed
        self.deadline_datetime = deadline_datetime
        self.idx = idx
        self.loss_rate = loss_rate
        self.num_bins = num_bins
        self.planning_fallacy_const = planning_fallacy_const
        self.rewards = rewards
        self.time_est = time_est
        self.unit_penalty = unit_penalty
        
        self.item_id = item_id
        if self.item_id is None:
            self.item_id = self.description
            
        self.latest_deadline_time = max(rewards.keys())
        self.optimal_reward = np.NINF

        self.time_transitions = {
            self.time_est: None
        }
        
        self.expected_loss = None  # TODO: Implement this
        self.values = deque()

    def __hash__(self):
        return id(self)

    def __str__(self):
        return f"Description: {self.description}\n" \
               f"Index: {self.idx}\n" \
               f"Latest deadline time: {self.latest_deadline_time}\n"  \
               f"Optimal reward: {self.optimal_reward}\n" \
               f"Time estimate: {self.time_est}\n"

    def compute_binning(self, num_bins=None):
        binned_distrib = get_binned_distrib(mu=self.time_est,
                                            num_bins=num_bins)
    
        bin_means = binned_distrib["bin_means"]
        bin_probs = binned_distrib["bin_probs"]
    
        self.time_transitions = dict()
    
        for i in range(len(bin_means)):
            mean = int(np.ceil(bin_means[i]))
            prob = bin_probs[i]
        
            self.time_transitions[mean] = prob

    def get_copy(self):
        return deepcopy(self)

    def get_deadline(self, t=0):
        if self.latest_deadline_time is None or t > self.latest_deadline_time:
            return None
        
        times = sorted(self.rewards.keys())
        return next(val for x, val in enumerate(times) if val >= t)

    def get_deadline_datetime(self):
        return self.deadline_datetime

    def get_description(self):
        return self.description
    
    def get_expected_loss(self):
        return self.expected_loss

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
        if beta == np.PINF:
            return 0
        
        if t > self.latest_deadline_time:
            
            if self.unit_penalty == np.PINF:
                return 0
            
            # Get discounted reward
            reward = discount * self.rewards[self.latest_deadline_time]
            
            return reward / (1 + beta * (t - self.latest_deadline_time))

        # Otherwise, get the reward for the next deadline that has been met
        deadline = self.get_deadline(t=t)
        
        # Get discounted reward that meets the deadline
        reward = self.rewards[deadline] * discount
        
        return reward / (1 + beta)

    def get_time_est(self):
        return self.time_est
    
    def get_time_transitions(self, t=None):
        if t is None:
            return self.time_transitions
        return self.time_transitions[t]
    
    def get_total_loss(self, cum_discount=None):
        # TODO: Find a better implementation.
        if cum_discount is None:
            return self.loss_rate * self.time_est
        return self.loss_rate * cum_discount
    
    def get_unit_penalty(self):
        return self.unit_penalty
    
    def get_values(self):
        return self.values
    
    def is_completed(self):
        return self.completed
    
    def set_completed(self):
        self.completed = True
    
    def set_description(self, description):
        self.description = description
        
    def set_expected_loss(self, expected_loss):
        self.expected_loss = expected_loss
    
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

    def set_time_est(self, time_est, num_bins=1):
        self.time_est = time_est
        
        # Compute time support
        self.compute_binning(num_bins=num_bins)
        
    def set_unit_penalty(self, unit_penalty):
        self.unit_penalty = unit_penalty
        

class Task(Item):
    """
    TODO:
        - Task predecessors/dependencies
    """
    
    def __init__(self, description, completed=False, deadline=None,
                 deadline_datetime=None, loss_rate=None, reward=0, task_id=None,
                 time_est=0, prob=1., today=False, unit_penalty=None):
        super().__init__(
            description=description,
            completed=completed,
            deadline_datetime=deadline_datetime,
            item_id=task_id,
            loss_rate=loss_rate,
            rewards={deadline: reward},
            time_est=time_est,
            unit_penalty=unit_penalty
        )
        
        self.deadline = deadline
        self.reward = reward
        self.goals = set()
        self.prob = prob
        self.today = today
        
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
        
    def get_goals(self):
        return self.goals
        
    def get_prob(self):
        return self.prob
    
    def get_total_reward(self):
        return self.get_reward() * self.time_est
    
    def is_today(self):
        return self.today

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
        
    def set_today(self, today):
        self.today = today
        
    def update_latest_deadline_time(self):
        self.latest_deadline_time = self.deadline
        
    def update_rewards_dict(self):
        self.rewards = {self.deadline: self.reward}


class Goal(Item):
    
    def __init__(self, description, completed=False, deadline_datetime=None,
                 gamma=None, goal_id=None, loss_rate=0, num_bins=1,
                 planning_fallacy_const=1., rewards=None, slack_reward=None,
                 tasks=None, unit_penalty=np.PINF):
        super().__init__(
            description=description,
            completed=completed,
            deadline_datetime=deadline_datetime,
            item_id=goal_id,
            loss_rate=loss_rate,
            num_bins=num_bins,
            planning_fallacy_const=planning_fallacy_const,
            rewards=rewards,
            time_est=0,
            unit_penalty=unit_penalty
        )
        
        self.slack_reward = slack_reward
        
        # Initialize task list
        self.gamma = gamma
        self.sorted_tasks_by_time_est = None
        self.sorted_tasks_by_deadlines = None
        
        self.tasks = tasks
        self.num_tasks = len(self.tasks)
        self.today_tasks = set()

        # Initialize 0-th state
        self.start_state = tuple()

        if tasks is not None:
            self.add_tasks(tasks)

        # Initialize policy, Q-value function and pseudo-rewards
        self.P = dict()  # {state: {time: action}}
        self.Q = dict()  # {state: {time: {action: {time': value}}}}
        self.PR = dict()  # Pseudo-rewards {(s, t, a): PR(s, t, a)}
        self.tPR = dict()  # Transformed PRs {(s, t, a): tPR(s, t, a)}

        self.F = dict()
        self.R = dict()
        self.log_prob = dict()

        # Initialize computations
        self.small_reward_pruning = 0
        self.already_computed_pruning = 0
        self.total_computations = 0

        # Initialize slack-off action
        self.slack_action = Task(
            f"This goal is sub-optimal! Please revise it!",
            deadline=np.PINF, reward=slack_reward, time_est=1)
        self.slack_action.set_idx(-1)
        self.slack_action.add_goal(self)

        # Initialize highest negative reward
        self.highest_negative_reward = np.NINF
        
        # Initialize dictionary of maximum future Q-values
        self.future_q = {
            0: None
        }
        
    def __str__(self):
        return super().__str__() + \
            f"Rewards: {self.rewards}\n" + \
            f"Slack reward: {self.slack_reward}\n"
            
    def add_tasks(self, tasks):
        
        self.start_state = list(0 for _ in range(self.num_tasks))
        
        for idx, task in enumerate(tasks):
            
            # If task has no deadline, set goal's deadline
            task.set_deadline(self.latest_deadline_time, compare=True)
            
            if task.get_loss_rate() is None:
                task.set_loss_rate(self.loss_rate)

            if task.get_num_bins() is None:
                task.set_num_bins(self.num_bins)

            if task.get_planning_fallacy_const() is None:
                task.set_planning_fallacy_const(self.planning_fallacy_const)

            if task.get_unit_penalty() is None:
                task.set_unit_penalty(self.unit_penalty)
                
            if task.is_today():
                self.today_tasks.add(idx)
                
            if task.is_completed():
                self.start_state[idx] = 1

            # TODO:
            #     - Pass number of bins
            #     - Pass planning fallacy
            
            # Connect task with goal
            task.add_goal(self)
            
            # Adjust task time estimate by the planning-fallacy constant
            task_time_est = ceil(task.get_time_est() *
                                 self.planning_fallacy_const)
            task.set_time_est(task_time_est, self.num_bins)
            
            # Add time estimate
            self.time_est += task_time_est
            
            # Set task index
            task.set_idx(idx)
            
        self.start_state = tuple(self.start_state)

        # Compute goal-level bins | # TODO: Increase to > 1 (!)
        self.compute_binning(num_bins=1)
        
        # Sorted list of tasks by time estimate
        self.sorted_tasks_by_time_est = deepcopy(tasks)
        self.sorted_tasks_by_time_est.sort(
            key=lambda item: (
                item.get_time_est(),
                item.get_deadline(),
                item.get_idx()
            )
        )
        
        # Sorted list of tasks by deadline (break ties with time estimate, idx)
        self.sorted_tasks_by_deadlines = deepcopy(self.sorted_tasks_by_time_est)
        self.sorted_tasks_by_deadlines.sort(
            key=lambda item: (
                item.get_deadline(),
                item.get_time_est(),
                item.get_idx()
            )
        )
        
        self.num_tasks = len(self.tasks)

    def check_missed_task_deadline(self, deadline, t):
        return t > deadline

    def compute_slack_reward(self, t=0):
        if self.slack_reward == 0:
            return 0
        
        if self.gamma < 1:
            cum_discount = ToDoList.get_cum_discount(t)
            return self.slack_reward * ((1 / (1 - self.gamma)) - cum_discount)
        
        return np.PINF

    def get_future_q(self, t=None):
        if t is not None:
            return self.future_q[t]
        return self.future_q

    def get_gamma(self):
        return self.gamma

    def get_highest_negative_reward(self):
        return self.highest_negative_reward

    def get_num_tasks(self):
        return self.num_tasks

    def get_policy(self, s=None, t=None):
        if s is not None:
            if t is not None:
                return self.P[s][t]
            return self.P[s]
        return self.P
    
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

    # TODO: Add comments (!)
    def get_start_state_pseudo_rewards(self, t=0):
        
        s = self.start_state
        
        best_a = -1
        best_q = self.compute_slack_reward(0)
        
        self.R[s][t][-1] = best_q
        
        for a, q in self.Q[s][t].items():
            if best_q <= q:
                best_a = a
                best_q = q
                
        loss = self.R[s][t][best_a]
        # next_q = best_q - loss

        PR = dict()
        gamma = ToDoList.get_discount(t)
        
        for a, q in self.Q[s][t].items():
            
            pr = gamma * q - best_q
            
            if np.isclose(pr, 0, atol=1e-6):
                pr = 0.

            PR[a] = pr

            if a == -1:
                task = self.slack_action
            else:
                task = self.tasks[a]
            
            task.set_optimal_reward(pr)
            
        return PR, best_a

    def get_slack_action(self):
        return self.slack_action

    def get_slack_reward(self):
        return self.slack_reward

    def get_start_state(self):
        return self.start_state

    def get_tasks(self):
        return self.tasks
    
    def set_future_q(self, t, value, compare=False):
        self.future_q.setdefault(t, value)
        if compare:
            self.future_q[t] = max(self.future_q[t], value)

    def set_gamma(self, gamma):
        self.gamma = gamma
        
    def set_slack_reward(self, slack_reward):
        self.slack_action.set_reward(slack_reward)
        self.slack_reward = slack_reward

    def solve(self, start_time=0, tasks=[], verbose=False):
        
        def solve_next_tasks(curr_state, mode, next_task=None, verbose=False):
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
            queue = None
            task_idx = None
            
            if next_task is None:
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
            else:
                task_idx = next_task.get_idx()

            if verbose:
                print(
                    f"\nCurrent task-level state: {s} | "
                    f"Current time: {t:>3d} | "
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

            # TODO: ...
            self.PR.setdefault(s, dict())
            self.PR[s].setdefault(t, dict())

            # TODO: ...
            self.F.setdefault(s, dict())
            self.F[s].setdefault(t, dict())

            if next_task is not None:
    
                """ Incorporate slack action for (state, time) """
                # Add slack reward to the dictionary
                slack_reward = self.compute_slack_reward(0)
                
                # Add slack reward in the Q-values
                self.Q[s][t].setdefault(-1, dict())
                self.Q[s][t][-1]["E"] = slack_reward
    
                # Add slack reward in the rewards
                self.R[s][t].setdefault(-1, dict())
                self.R[s][t][-1]["E"] = slack_reward
    
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
                time_transitions = next_task.get_time_transitions().items()
                
                if verbose:
                    print(f"Time estimates: {time_transitions}")
                    
                for time_est, prob_t_ in time_transitions:
                
                    # Make time transition
                    t_ = t + time_est

                    # Initialize Q-values for (t)ime'
                    self.Q[s_].setdefault(t_, dict())

                    # Increase total-computations counter
                    self.total_computations += 1

                    # Get cumulative discount w.r.t. task duration
                    cum_discount = ToDoList.get_cum_discount(time_est)
                    
                    # Calculate total loss for next action (immediate "reward")
                    r = next_task.get_total_loss(cum_discount=cum_discount)
                    
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
                        next_state["log_prob"] += np.log(prob_t_)
                        
                        # Store log probability of the next state
                        self.log_prob.setdefault(s_, dict())
                        self.log_prob[s_][t_] = next_state["log_prob"]
                        
                        # Add deadline to the missed deadlines if not attained
                        if self.check_missed_task_deadline(task_deadline, t_):
                            
                            # Compute total penalty for missing task deadline
                            total_penalty = next_task.get_unit_penalty() * \
                                            (t_ - task_deadline)
                            
                            next_state["penalty_factor"] += total_penalty

                        if idx is None:
                            idx = -1
                            
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
                        total_reward = r + ToDoList.get_discount(time_est) * r_
    
                        # If total reward is negative, compare it with the
                        # highest negative reward and substitute if higher
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

                """ Add absorbing terminal state """
                # self.Q[s].setdefault(np.PINF, dict())
                # self.Q[s][np.PINF].setdefault(None, dict())
                # self.Q[s][np.PINF][None]["E"] = 0

                # self.R[s].setdefault(np.PINF, dict())
                # self.R[s][np.PINF].setdefault(None, dict())
                # self.R[s][np.PINF][None]["E"] = 0

                # TODO: ...
                self.PR[s].setdefault(np.PINF, dict())
                self.F[s].setdefault(np.PINF, dict())

                # Get total penalty factor
                beta = curr_state["penalty_factor"]
                
                # Compute reward
                term_value = self.get_reward(beta=beta, t=t, discount=1.)

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
                # if "E" in self.Q[s][t][None].keys():
                #
                #     self.Q[s][t][None]["E"] = \
                #         max(self.Q[s][t][None]["E"], term_value)
                #
                # # Otherwise, initialize it with the current Q value
                # else:
                #     self.Q[s][t][None]["E"] = term_value

                # Compute Q-values for (s)tate and (t)ime
                self.Q.setdefault(s, dict())
                self.Q[s].setdefault(t, dict())
                self.Q[s][t].setdefault(None, dict())
                self.Q[s][t][None]["E"] = term_value
            
        # Initialize starting state and time
        s = tuple(0 for _ in range(self.num_tasks))
        t = start_time
        
        # Initialize log probability
        self.log_prob.setdefault(s, dict())
        self.log_prob[s][t] = 0
        
        # Initialize state
        curr_state = {
            "s": s,
            "t": t,
            "idx_deadlines": 0,
            "idx_time_est":  0,
            "penalty_factor": 0.,
            "log_prob": 0,
        }

        # Take next action to be from the task list sorted w.r.t. time estimates
        solve_next_tasks(curr_state, mode="time_est", verbose=verbose)
        
        # Take next action to be from the task list sorted w.r.t. deadlines
        solve_next_tasks(curr_state, mode="deadline", verbose=verbose)
        
        for task in tasks:
            # Take next action to be from the task list sorted w.r.t. time estimates
            solve_next_tasks(curr_state, mode="time_est",
                             next_task=task, verbose=verbose)
    
            # Take next action to be from the task list sorted w.r.t. deadlines
            solve_next_tasks(curr_state, mode="deadline",
                             next_task=task, verbose=verbose)

        # Get optimal action and value for the starting state and time
        a, r = ToDoList.max_from_dict(self.Q[s][t])

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

    def solve_chain(self, available_time=np.PINF, start_time=0, verbose=False):
        
        def solve_branching(curr_state, next_task=None, verbose=False):
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
    
            queue = self.sorted_tasks_by_deadlines
            task_idx = None
            
            if next_task is None:
                
                # Get next uncompleted task (if one exists)
                while idx < len(queue):
                    
                    # Get next task from the queue
                    task = queue[idx]
                    
                    # Get task index
                    task_idx = task.get_idx()
                    
                    # If the next task in the queue is completed
                    if s[task_idx]:
                        idx += 1
                    else:
                        next_task = task
                        break
                        
            else:
                # Set action to be the index of the next task
                task_idx = next_task.get_idx()
    
            if verbose:
                print(
                    f"\nCurrent task-level state: {s} | "
                    f"Current time: {t:>3d} | "
                    f"Task index: {task_idx if task_idx is not None else '-'} | "
                    f"Idx: {idx} | "
                    ,
                    end=""
                )

            if next_task is not None:
    
                # Set action to be the index of the next task
                a = next_task.get_idx()
        
                # Get next state by taking (a)ction in (s)tate
                s_ = ToDoList.exec_action(s, a)
        
                # Get the latest deadline time of the next tasks
                task_deadline = next_task.get_latest_deadline_time()
        
                # Get time transitions of the next state
                time_transitions = next_task.get_time_transitions().items()
        
                # Initialize Q-value
                q = 0
                
                # Initialize expected reward
                exp_reward = 0
        
                for time_est, prob_t_ in time_transitions:
            
                    # Make time transition
                    t_ = t + time_est
            
                    # Increase total-computations counter
                    self.total_computations += 1

                    # Get cumulative discount w.r.t. task duration
                    cum_discount = ToDoList.get_cum_discount(time_est)
            
                    # Calculate total loss for next action (immediate "reward")
                    r = next_task.get_total_loss(cum_discount=cum_discount)
            
                    # Generate next task-level state (by updating current state)
                    next_state = deepcopy(curr_state)
                    next_state["s"] = s_
                    next_state["t"] = t_
                    next_state["idx"] = idx
            
                    # Add deadline to the missed deadlines if not attained
                    if self.check_missed_task_deadline(task_deadline, t_):
                        
                        # Compute total penalty for missing task deadline
                        total_penalty = next_task.get_unit_penalty() * \
                                        (t_ - task_deadline)
                
                        # Update penalty for the next state
                        next_state["penalty_factor"] += total_penalty
            
                    # Solve branching of the next state
                    q_, _ = solve_branching(next_state, verbose=verbose)
            
                    # Update expected reward for (state, time, action)
                    exp_reward += prob_t_ * r
            
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
                    #     total_reward = q + self.gamma * q_
            
                    # Compute discount factor
                    gamma = ToDoList.get_discount(time_est)
            
                    # Add contribution to the expected value of taking (a)ction
                    q += prob_t_ * (r + gamma * q_)
                    
                # Return Q-value of the current state
                return q, exp_reward
    
            # ===== Terminal state =====
            else:
                
                # Get total penalty factor
                beta = curr_state["penalty_factor"]
        
                # Compute reward
                term_value = self.get_reward(beta=beta, t=t, discount=1.)
                
                if verbose:
                    print(f"Beta: {beta}")
                
                # Return (potentially discounted) goal reward
                return term_value, 0
        
        # Initialize starting (s)tate and (t)ime
        s = self.start_state
        t = start_time
        
        # Initialize entries
        self.Q.setdefault(s, dict())
        self.R.setdefault(s, dict())
        
        self.Q[s].setdefault(t, dict())
        self.R[s].setdefault(t, dict())

        """ Incorporate slack action for (state, time) """
        # Add slack reward to the dictionary
        slack_reward = self.compute_slack_reward(0)

        # Add slack reward in the Q-values & rewards
        self.Q[s][t].setdefault(-1, slack_reward)
        self.R[s][t].setdefault(-1, slack_reward)

        # TODO: Comment...
        tasks_to_iterate = deque()
        
        if t == 0:
            
            if available_time == np.PINF or available_time > self.time_est:
                
                # Add all tasks
                tasks_to_iterate.extend(self.sorted_tasks_by_deadlines)
                
                # Set index as list was completely iterated
                self.today_tasks = set(range(self.num_tasks))
                
            else:
                
                for idx in self.today_tasks:
                    
                    tasks_to_iterate.append(self.tasks[idx])
                
                # Reduce the number of tasks for which this has been iterated
                
                available_time = deepcopy(available_time)
    
                for task in self.sorted_tasks_by_deadlines:
                    
                    idx = task.get_idx()
                    
                    if idx not in self.today_tasks:
                    
                        time_est = task.get_time_est()
                        
                        tasks_to_iterate.append(task)
                        
                        available_time -= time_est
                        
                        self.today_tasks.add(idx)
                        
                        if available_time < 0:
                            break
                
        else:
            tasks_to_iterate.append(self.sorted_tasks_by_deadlines[0])
            
        # Append least-optimal action in order to get lower bound on f(s, a)
        last_task = self.sorted_tasks_by_deadlines[-1]

        if last_task.get_idx() not in self.today_tasks:

            tasks_to_iterate.append(last_task)

            self.today_tasks.add(last_task.get_idx())
            
        tic = time.time()
            
        for task in tasks_to_iterate:
            
            # Initialize state
            curr_state = {
                "s":              deepcopy(s),
                "t":              t,
                "idx":            0,
                "penalty_factor": 0.,
            }
    
            # Get index of the first action
            a = task.get_idx()
            
            # If action is not completed
            if not s[a]:
                
                if verbose:
                    print(f"\n===== Starting action: {a} =====", end="")
                    
                # Solve branching procedure
                q, exp_reward = solve_branching(curr_state, next_task=task,
                                                verbose=verbose)
                
                # Store Q-value and immediate reward/loss for (a)ction
                self.Q[s][t][a] = q
                self.R[s][t][a] = exp_reward
                
        toc = time.time()

        if verbose and t == 0:
            print(
                f"{t:>3d} | "
                f"Number of tasks: {len(tasks_to_iterate)} | "
                f"Time: {toc - tic:.2f}\n"
            )
        
        # Return maximum Q-value
        return max(self.Q[s][t].values())
    
    
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
        
        self.description = "To-do list"  # TODO: Change name (?)
        self.goals = goals
        self.end_time = end_time
        self.gamma = gamma
        self.slack_reward = slack_reward
        self.start_time = start_time

        # Slack-off action
        self.slack_action = Task(
            f"{self.description} + slack-off action", deadline=np.PINF,
            reward=self.slack_reward, time_est=1)
        self.slack_action.set_idx(-1)

        # Set number of goals
        self.num_goals = len(self.goals)

        # Calculate total time estimate of the to-do list
        self.total_time_est = 0
        
        # Add goals to the to-do list
        self.add_goals(self.goals)
        
        # "Cut" horizon in order to reduce the number of computations
        self.end_time = min(self.end_time, self.total_time_est)
        
        # Generate discounts | TODO: Add epsilon as input parameter (!)
        ToDoList.generate_discounts(epsilon=0., gamma=self.gamma,
                                    horizon=self.end_time, verbose=False)
        
        # Initialize policy, Q-value function and pseudo-rewards
        self.P = dict()  # Optimal policy {state: action}
        self.Q = dict()  # Action-value function {state: {action: value}}
        self.PR = dict()  # Pseudo-rewards {(s, t, a): PR(s, t, a)}
        self.tPR = dict()  # Transformed PRs {(s, t, a): tPR(s, t, a)}
        
        self.F = dict()
        self.R = dict()
        self.log_prob = dict()

        # Initialize computation counters
        self.small_reward_pruning = 0
        self.already_computed_pruning = 0
        self.total_computations = 0
        
        # Initialize highest negative reward
        self.highest_negative_reward = np.NINF
        
        # Initialize 0-th state
        self.start_state = tuple(0 for _ in range(self.num_goals))
        
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
        max_q = np.NINF
        
        for a in d.keys():
            
            # Get expected Q-value
            q = d[a]["E"]
            
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
        for idx, goal in enumerate(goals):
        
            # Increase total time estimate of the to-do list
            self.total_time_est += goal.get_time_est()
        
            # Set discount parameter for goal if not defined
            if goal.get_gamma() is None:
                goal.set_gamma(self.gamma)
        
            # Set slack reward for goal's slack-off action if not defined
            if goal.get_slack_reward() is None:
                goal.set_slack_reward(self.slack_reward)
                
            # TODO: Completed goals in start_state (?)
                
            # Set goal index
            goal.set_idx(idx)

    def compute_slack_reward(self, t=0):
        if self.slack_reward == 0:
            return 0
        
        if self.gamma < 1:
            cum_discount = ToDoList.get_cum_discount(t)
            return self.slack_reward * ((1 / (1 - self.gamma)) - cum_discount)
        
        return np.PINF

    def get_description(self):
        return self.description

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

    def get_q_values(self, s=None, t=None, a=None):
        if s is not None:
            if t is not None:
                if a is not None:
                    return self.Q[s][t][a]
                return self.Q[s][t]
            return self.Q[s]
        return self.Q

    def get_start_state(self):
        return self.start_state

    def get_start_time(self):
        return self.start_time
    
    def get_slack_action(self):
        return self.slack_action

    def get_slack_reward(self):
        return self.slack_reward

    def set_gamma(self, gamma):
        assert 0 < gamma <= 1
        self.gamma = gamma

    def set_slack_reward(self, slack_reward):
        self.slack_action.set_reward(slack_reward)
        self.slack_reward = slack_reward

    def solve(self, available_time=np.PINF, start_time=0, verbose=False):
    
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

            # Initialize policy entries for (s)tate and ((s)tate, (t)ime))
            self.P.setdefault(s, dict())
            self.P[s].setdefault(t, dict())

            # Initialize Q-function entry for the (state, time) pair
            self.Q.setdefault(s, dict())
            self.Q[s].setdefault(t, dict())

            # Initialize reward entries for (s)tate and ((s)tate, (t)ime))
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

                    # TODO: Probability of transition to next state
                    prob = 1

                    # Store log probability of the next state
                    self.log_prob.setdefault(s_, dict())
                    self.log_prob[s_][t_] = np.log(prob)

                    # Calculate total reward for next action
                    # result = next_goal.solve(start_time=t, verbose=verbose)
                    # r = result["r"]
                    
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

                        r = self.R[s][t][a]["E"]
                        
                    # Explore the next goal-level state
                    else:
    
                        # Initialize entry for (state, time, action)
                        self.R[s][t].setdefault(a, dict())
                        self.R[s][t][a].setdefault("E", 0)
    
                        r = next_goal.solve_chain(available_time=available_time,
                                                  start_time=t, verbose=verbose)
    
                        self.R[s][t][a][t_] = r
                        self.R[s][t][a]["E"] += prob * r
    
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
                    
                    # Store future Q-value
                    next_goal.set_future_q(t_, r_, compare=True)
                    
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
                
                # Initialize dictionary for the terminal state Q-value
                self.Q[s][t].setdefault(None, dict())
                self.Q[s][t][None][np.PINF] = 0
                self.Q[s][t][None]["E"] = 0

                # Compute reward for reaching terminal state s in time t
                self.R[s][t].setdefault(None, dict())
                self.R[s][t][None][np.PINF] = 0
                self.R[s][t][None]["E"] = 0

        # Iterate procedure
        s = tuple(0 for _ in range(self.num_goals))
        t = start_time

        # Initialize log probability
        self.log_prob.setdefault(s, dict())
        self.log_prob[s][t] = 0

        curr_state = {
            "s": s,
            "t": t
            # TODO: log_prob
        }

        # Start iterating
        solve_next_goals(curr_state, verbose=verbose)
        
        # Get best (a)ction in the start state and its corresponding (r)eward
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
