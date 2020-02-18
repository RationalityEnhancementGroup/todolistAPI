import itertools
import numpy as np
import random
import time

from todolistMDP import mdp


class Task:
    def __init__(self, description, completed=False, goal=None,
                 prob=1., reward=0, task_id=None, time_est=1):
        """
        # TODO: reward=None
        # TODO: Complete this...

        Non-goal tasks are goals for themselves with reward proportional to the
        value of the misc goal node, and infinite deadline. To simulate infinite
        deadline, we set the deadline of each non-goal task to be an addition to
        the latest deadline of all "real" goals, i.e.
            latest_deadline(all_goals) + sum(non_goal_task["time_est"])

        Args:
            description: Description of the task
            time_est: Units of time required to perform a task

            completed: Whether the task has been completed
            goal: Goal object whom this task belongs to
            prob: Probability of successful completion of the task
            reward: Reward of completing the task, usually 0 (?!)
        """
        
        # Set parameters
        self.description = description
        
        self.task_id = task_id
        if self.task_id is None:
            self.task_id = description
        
        self.completed = completed
        self.goal = goal
        self.prob = prob
        self.reward = reward
        self.time_est = time_est
    
    def __str__(self):
        return f'Description: {self.description}\n' \
               f'Completed: {self.completed}\n' \
               f'ID: {self.task_id}\n' \
               f'Goal: {self.goal.get_description()}\n' \
               f'Probability: {self.prob}\n' \
               f'Reward: {self.reward}\n' \
               f'Time est.: {self.time_est}\n'
    
    def get_copy(self):
        return Task(self.description, completed=self.completed, goal=self.goal,
                    prob=self.prob, reward=self.reward, time_est=self.time_est)
    
    def get_description(self):
        return self.description
    
    def get_goal(self):
        return self.goal
    
    def get_prob(self):
        return self.prob
    
    def get_reward(self):
        return self.reward
    
    def get_task_id(self):
        return self.task_id
    
    def get_time_est(self):
        return self.time_est
    
    def is_completed(self):
        return self.completed
    
    # def set_completed(self, completed):
    #     if self.completed != completed:
    #         self.completed = completed
    #         if completed:
    #             self.goal.add_completed_time(self.time_est)
    #             # TODO: Check whether this is the last completed task
    #         else:
    #             self.goal.add_completed_time(-self.time_est)
    #             # TODO: Check whether this makes a completed goal active again
    
    def set_goal(self, goal):
        if self.goal is not goal:
            
            # Remove task from the old goal
            if self.goal is not None:
                self.goal.remove_task(self)
            
            # Set new goal
            self.goal = goal
            
            # Add task to the new goal
            self.goal.add_task(self)
    
    def set_reward(self, value):
        self.reward = value


class Goal:
    def __init__(self, description, rewards, tasks,
                 completed=False, goal_id=None, penalty=0):
        """
        # TODO: Complete this...
        
        Non-goal tasks are goals for themselves with reward proportional to the
        value of the misc goal node, and infinite deadline. To simulate infinite
        deadline, we set the deadline of each non-goal task to be an addition to
        the latest deadline of all "real" goals, i.e.
            latest_deadline(all_goals) + sum(non_goal_task["time_est"])
        
        Args:
            description: String description of the goal
            rewards: {Time of completion: Reward}
            tasks: [Task]
            
            completed: Whether it has been completed
            penalty: Penalty points for failing to meet the deadline
        """
        # Parameters
        self.description = description
        self.rewards = rewards
        
        self.goal_id = goal_id
        if self.goal_id is None:
            self.goal_id = description
        
        # Set up a deadline
        self.latest_deadline = max(rewards.keys())

        self.completed = completed
        self.penalty = penalty
        
        # Calculate time and value estimation
        self.completed_time_est = 0  # Time estimation of completed tasks
        self.uncompleted_time_est = 0  # Time estimation of uncompleted tasks
        self.total_time_est = 0  # Time estimation of all tasks

        # Split tasks into completed and uncompleted
        self.all_tasks = tasks
        self.completed_tasks = set()
        self.uncompleted_tasks = set()

        for task in self.all_tasks:
            task.set_goal(self)  # Set a reference from the tasks to the goal

            if task.completed:
                self.completed_tasks.add(task)
                self.completed_time_est += task.get_time_est()
            else:
                self.uncompleted_tasks.add(task)
                self.uncompleted_time_est += task.get_time_est()
                
        self.update_total_time_est()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self.get_latest_deadline_time() == other.get_latest_deadline_time()
    
    def __ne__(self, other):
        return self.get_latest_deadline_time() != other.get_latest_deadline_time()

    def __ge__(self, other):
        return self.get_latest_deadline_time() >= other.get_latest_deadline_time()

    def __gt__(self, other):
        return self.get_latest_deadline_time() > other.get_latest_deadline_time()

    def __le__(self, other):
        return self.get_latest_deadline_time() <= other.get_latest_deadline_time()

    def __lt__(self, other):
        return self.get_latest_deadline_time() < other.get_latest_deadline_time()

    def __str__(self):
        return f'Description: {self.description}\n' \
               f'Rewards: {self.rewards}\n' \
               f'Completed: {self.completed}\n' \
               f'ID: {self.goal_id}\n' \
               f'Latest deadline: {self.latest_deadline}\n' \
               f'Total time est.: {self.total_time_est}\n'

    def get_completed_tasks(self):
        return self.completed_tasks
    
    def get_completed_time_est(self):
        return self.completed_time_est

    def get_deadline_penalty(self):
        return self.penalty

    def get_description(self):
        return self.description

    def get_goal_id(self):
        return self.goal_id
    
    def get_latest_deadline_time(self):
        return self.latest_deadline

    def get_tasks(self):
        return self.all_tasks

    def get_total_time_est(self):
        return self.total_time_est

    def get_uncompleted_tasks(self):
        return self.uncompleted_tasks

    def get_uncompleted_time_est(self):
        return self.uncompleted_time_est

    def get_reward(self, time):
        """
        
        Args:
            time:

        Returns:
            Return reward based on time
        """
        # If the latest deadline has not been met, get no reward
        if time > self.get_latest_deadline_time():
            return 0

        # Otherwise, get the reward for the next deadline that has been met
        times = sorted(self.rewards.keys())
        t = next(val for x, val in enumerate(times) if val >= time)
        
        return self.rewards[t]

    def get_reward_dict(self):
        return self.rewards

    def is_completed(self, check_tasks=False):
        """
        Method for checking whether a goal is complete by checking
        if all tasks are complete
        
        Args:
            check_tasks: Whether to check whether all tasks are completed
                         or to return the cached value
        
        Returns:
            Completion status
        """
        if check_tasks:
            for task in self.all_tasks:
                if not task.is_completed():
                    return False
            # self.set_completed(True)
        return self.completed

    def add_completed_time(self, time_est):
        self.completed_time_est += time_est
        self.uncompleted_time_est -= time_est
        self.update_total_time_est()

    def add_task(self, task):
        if self.all_tasks is None:
            self.all_tasks = []
        
        if task.get_goal() is not self:
            self.all_tasks.append(task)
            task.set_goal(self)
            
            task_time_est = task.get_time_est()
            
            if not task.completed:
                # self.set_completed(False)
                self.uncompleted_time_est += task_time_est
            else:
                self.completed_time_est += task_time_est
            
            self.total_time_est += task_time_est
            # self.value_est += task.get_prob() * task.get_reward()

    # def remove_task(self, task):
    #     if task in self.all_tasks:
    #         self.all_tasks.remove(task)
    #         task_time_est = task.get_time_est()
    #
    #         # Subtract task time estimation
    #         if task.is_completed():
    #             self.completed_time_est -= task_time_est
    #         else:
    #             self.uncompleted_time_est -= task_time_est
    #
    #         self.total_time_est -= task_time_est
    #
    #         # Subtract task value
    #         self.value_est -= task.get_prob() * task.get_reward()
    
    # def reset_completed(self):
    #     self.completed = False
    #     self.uncompleted_time_est = 0
    #     self.completed_time_est = 0
    #
    #     for task in self.all_tasks:
    #         task.set_completed(False)
    #         self.uncompleted_time_est += task.get_time_est()
    #
    #     self.update_total_time_est()

    # def set_completed(self, completed):
    #     self.completed = completed
    #     if completed:
    #         for task in self.all_tasks:
    #             task.set_completed(completed)
    #
    #         # Change time-estimation values
    #         self.completed_time_est = self.total_time_est
    #         self.uncompleted_time_est = 0
    #         self.update_total_time_est()
            
    def update_total_time_est(self):
        self.total_time_est = self.completed_time_est + \
                              self.uncompleted_time_est
        
    def set_rewards_dict(self, rewards):
        self.rewards = rewards
        self.latest_deadline = max(rewards.keys())

    def scale_uncompleted_task_time(self, scale, up=True):
        if up:
            self.uncompleted_time_est = self.uncompleted_time_est * scale
        else:
            self.uncompleted_time_est = self.uncompleted_time_est // scale


class ToDoList:
    def __init__(self, goals, start_time=0, end_time=None):
        """
        
        Args:
            goals: List of all goals
            start_time:  # TODO: Remove this?!
            end_time:  # TODO: Remove this?!
        """
        # Goals
        self.goals = goals  # TODO: Change list to dictionary
        self.completed_goals = set()
        self.uncompleted_goals = set()
        
        self.all_tasks = set()  # TODO: Change list to dictionary
        self.completed_tasks = set()
        self.uncompleted_tasks = set()
        
        self.time = start_time  # Current time  | TODO: Do we need this?
        self.start_time = start_time  # TODO: Do we need this?
        self.end_time = end_time  # TODO: Do we need this?
        
        self.max_deadline = float('-inf')  # TODO: Replace -> max(goal_deadline)

        # Add goals and tasks to the to-do list
        for goal in self.goals:
            if goal.is_completed():
                self.completed_goals.add(goal)
            else:
                self.uncompleted_goals.add(goal)

            # Split tasks into completed and uncompleted
            for task in goal.get_tasks():
                self.all_tasks.add(task)  # TODO: goal.get_tasks
    
                if task.is_completed():
                    # TODO: goal.get_completed_tasks
                    self.completed_tasks.add(task)
                else:
                    # TODO: goal.get_uncompleted_tasks
                    self.uncompleted_tasks.add(task)
                    
            self.max_deadline = max(self.max_deadline, goal.get_latest_deadline_time())
            
        if self.end_time is None:
            self.end_time = self.max_deadline + 1  # TODO: Why + 1?
            
        # Tasks | # TODO: Move this to the Goal class
        # for goal in self.goals:
        #     self.tasks.extend(goal.get_tasks())
        # self.completed_tasks = set([task for task in self.tasks
        #                             if task.is_complete()])
        # self.uncompleted_tasks = set([task for task in self.tasks
        #                               if not task.is_complete()])
        
    def __str__(self):
        return f'Current Time: {str(self.time)}\n' \
               f'Goals: {str(self.goals)}\n' \
               f'Completed Goals: {str(self.completed_goals)}\n' \
               f'"Tasks: {str(self.all_tasks)}\n' \
               f'Completed Tasks: + {str(self.completed_tasks)}\n'

    def action(self, task=None):
        """
        Do a specified action
        
        Args:
            task: ...; If not defined, it completes a random task

        Returns:

        """
        # TODO: Randomly get an uncompleted task from an uncompleted goal
        if task is None:
            task = random.sample(self.uncompleted_tasks, 1)[0]
        
        reward = 0
        prev_time = self.time
        curr_time = self.time + task.get_time_est()
        self.increment_time(task.get_time_est())
        
        reward += self.do_task(task)
        reward += self.check_deadlines(prev_time, curr_time)
        
        return reward

    def check_deadlines(self, prev_time, curr_time):
        """
        Check which goals passed their deadline between prev_time and curr_time
        If goal passed deadline during task, incur penalty
        """
        penalty = 0

        for goal in self.uncompleted_goals:
            # Check:
            # 1) goal is now passed deadline at curr_time
            # 2) goal was not passed deadline at prev_time

            # TODO: Shouldn't we have an inequality in one of the tests?!
            if curr_time > goal.get_latest_deadline_time() and \
                    not prev_time > goal.get_latest_deadline_time():
                penalty += goal.get_deadline_penalty()
                
        return penalty
    
    def do_task(self, task):
        # TODO: Change this so that it is on a Goal level (!)
        goal = task.get_goal()
        threshold = task.get_prob()
        
        reward = task.get_reward()
        p = random.random()
        
        # Check whether the task is completed on time
        # TODO: self.time + task.get_time_cost() (!?)
        if p < threshold and self.time <= goal.get_latest_deadline_time() \
                and not goal.is_completed():
                
            task.set_completed(True)
            self.uncompleted_tasks.discard(task)
            self.completed_tasks.add(task)
            
            # If completion of the task completes the goal
            if goal.is_completed():
                reward += goal.get_reward(self.time)  # Goal completion reward
                self.uncompleted_goals.discard(goal)
                self.completed_goals.add(goal)
                
        return reward

    # ===== Getters =====
    
    def get_end_time(self):
        return self.end_time
    
    # TODO: get_goal

    def get_goals(self):
        return self.goals

    # TODO: Remove this?!
    def get_all_tasks(self):
        return self.all_tasks

    def get_time(self):
        return self.time

    # ===== Setters =====
    def increment_time(self, time_est=1):
        self.time += time_est

    def add_goal(self, goal):
        """
        Add an entire goal
        """
        self.goals.append(goal)
        self.all_tasks.extend(goal.get_tasks())  # TODO: Fix this...
        
        if goal.is_completed():
            self.completed_goals.add(goal)
        else:
            self.uncompleted_goals.add(goal)

    # TODO: Duplicated @ Goal task?!
    def add_task(self, goal, task):
        """
        Adds task to the specified goal
        """
        self.all_tasks.append(task)  # TODO: Fix this...
        goal.add_task(task)


class ToDoListMDP(mdp.MarkovDecisionProcess):
    """
    State: (boolean vector for task completion, time)
    """

    def __init__(self, to_do_list, gamma=1.0, living_reward=0.0, noise=0.0):
        # Parameters
        self.gamma = gamma
        self.living_reward = living_reward
        self.noise = noise

        # To-do list
        self.to_do_list = to_do_list
        self.start_state = self.get_start_state()

        # Create mapping of indices to tasks represented as list
        self.index_to_task = list(to_do_list.get_all_tasks())
        
        # Create mapping of tasks to indices represented as dict
        self.task_to_index = {}
        for i, task in enumerate(self.index_to_task):
            self.task_to_index[task] = i

        # Creating Goals and their corresponding tasks pointers
        self.goals = self.to_do_list.get_goals()
        self.goal_to_indices = {}
        for goal in self.goals:
            self.goal_to_indices[goal] = [self.task_to_index[task]
                                          for task in goal.get_tasks()]

        # Generate state space
        self.states = []
        num_tasks = len(to_do_list.get_all_tasks())
        for t in range(self.to_do_list.get_end_time() + 2):  # TODO: Why +2?!
            for bit_vector in itertools.product([0, 1], repeat=num_tasks):
                state = (bit_vector, t)
                self.states.append(state)

        # Mapping from (binary vector x time) to integer (?!)
        # TODO: Potentially unnecessary...
        # self.state_to_index = {self.states[i]: i
        #                        for i in range(len(self.states))}

        self.reverse_DAG = MDPGraph(self)
        self.linearized_states = self.reverse_DAG.linearize()

        self.v_states = {}
        self.optimal_policy = {}
        
        # Pseudo-rewards
        self.pseudo_rewards = {}  # {(s, a, s') --> PR(s, a, s')}
        self.transformed_pseudo_rewards = {}  # {(s, a, s') --> PR'(s, a, s')}
        # self.calculate_pseudo_rewards()  # Calculate PRs for each state
        # self.transform_pseudo_rewards()  # Apply linear transformation to PR'

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

    @staticmethod
    def tasks_to_binary(tasks):
        """
        Convert a list of Task objects to a bit vector with 1 being complete and
        0 if not complete.
        """
        return tuple([1 if task.is_completed() else 0 for task in tasks])

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

    def get_gamma(self):
        return self.gamma

    def is_goal_completed(self, goal, state):
        """
        Given a Goal object and current state
        Check if the goal is completed
        """
        tasks = state[0]
    
        for i in self.goal_to_indices[goal]:
            if tasks[i] == 0:
                return False
    
        return True

    def get_linearized_states(self):
        return self.linearized_states

    def get_optimal_policy(self, state=None):
        """
        Returns the mapping of state to the optimal policy at that state
        """
        if state is not None:
            return self.optimal_policy[state]
        return self.optimal_policy

    def get_pseudo_rewards(self, transformed=False):
        """ getter method for pseudo-rewards
        pseudo_rewards is stored as a dictionary,
        where keys are tuples (s, s') and values are PR'(s, a, s')
        """
        if transformed:
            return self.transformed_pseudo_rewards
        return self.pseudo_rewards

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
        goal = task.get_goal()
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
                reward += goal.get_deadline_penalty()
    
        return reward

    def get_start_state(self):
        """
        Return the start state of the MDP.
        """
        # TODO: I don't think that this is the start state necessarily because
        #        it returns the current state with time 0...
        start_state = self.tasks_to_binary(self.to_do_list.get_all_tasks())
        return start_state, 0  # TODO: Maybe curr_state, self.get_time()

    # def get_state_index(self, state):
        # TODO: Potentially unnecessary function...
        # return self.state_to_index[state]

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

    def is_task_active(self, task, time):
        """
        Check if the goal for a given task is still active at a time
        """
        goal = task.get_goal()
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

    # ===== Setters =====
    def set_living_reward(self, reward):
        """
        The (negative) reward for exiting "normal" states.
        Note that in the R+N text, this reward is on entering
        a state and therefore is not clearly part of the state's
        future rewards.
        """
        self.living_reward = reward

    def set_noise(self, noise):
        """
        The probability of moving in an unintended direction.
        """
        self.noise = noise


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
        print(f'Time elapsed: {end - start} seconds.\n')
        
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
