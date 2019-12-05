import itertools
import numpy as np
import random
import time

from todolistMDP import mdp


class Goal:
    def __init__(self, description, tasks, reward,
                 completed=False, deadline=None, non_goal=False, penalty=0):
        """
        
        Args:
            description: String description of the goal
            reward: {Time of completion: Reward}
            tasks: [Task]
            
            completed: Whether it has been completed
            deadline: Latest deadline time
            non_goal: Whether it is a non-goal
            penalty: Penalty points for failing to meet the deadline
        """
        # Parameters
        self.description = description
        self.reward = reward
        self.tasks = tasks
        for task in self.tasks:
            task.set_goal(self)  # Reference from the tasks to the goal

        # Set up a deadline
        self.deadline = deadline
        if self.deadline is None:
            self.deadline = max(reward.keys())

        self.completed = completed
        self.non_goal = non_goal
        self.penalty = penalty
        
        # Calculate time estimation
        self.completed_time_est = 0  # Time estimation of completed tasks
        self.uncompleted_time_est = 0  # Time estimation of uncompleted tasks
        self.total_time_est = 0  # Time estimation of all tasks
        
        for task in self.tasks:
            if task.completed:
                self.completed_time_est += task.get_time_est()
            else:
                self.uncompleted_time_est += task.get_time_est()
        self.update_total_time_est()

        # Calculate value_estimation
        self.value_est = 0
        for task in self.tasks:
            self.value_est += task.get_prob() * task.get_reward()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self.get_deadline_time() == other.get_deadline_time()
    
    def __ne__(self, other):
        return self.get_deadline_time() != other.get_deadline_time()

    def __ge__(self, other):
        return self.get_deadline_time() >= other.get_deadline_time()

    def __gt__(self, other):
        return self.get_deadline_time() > other.get_deadline_time()

    def __le__(self, other):
        return self.get_deadline_time() <= other.get_deadline_time()

    def __lt__(self, other):
        return self.get_deadline_time() < other.get_deadline_time()

    def __str__(self):
        return f'Description: {self.description}\n' \
               f'Reward: {self.reward}\n' \
               f'Completed: {self.completed}\n' \
               f'Latest deadline: {self.deadline}\n' \
               f'Total time est.: {self.total_time_est}\n'

    def get_completed_time_est(self):
        return self.completed_time_est

    def get_deadline_penalty(self):
        return self.penalty

    def get_deadline_time(self):
        return self.deadline

    def get_description(self):
        return self.description

    def get_total_time_est(self):
        return self.total_time_est

    def get_tasks(self):
        return self.tasks

    def get_uncompleted_time_est(self):
        return self.uncompleted_time_est

    def get_reward(self, time):
        """
        
        Args:
            time:

        Returns:
            Return reward based on time
        """
        if time > self.get_deadline_time():
            return 0
        times = sorted(self.reward.keys())
        t = next(val for x, val in enumerate(times) if val >= time)
        return self.reward[t]

    def get_reward_dict(self):
        return self.reward

    def is_complete(self, check_tasks=False):
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
            for task in self.tasks:
                if not task.is_complete():
                    return False
            self.set_completed(True)
        return self.completed

    def is_non_goal(self):
        return self.non_goal

    def add_completed_time(self, time_est):
        self.completed_time_est += time_est
        self.uncompleted_time_est -= time_est
        self.update_total_time_est()

    def add_task(self, task):
        self.tasks.append(task)
        task.set_goal(self)
        if not task.completed:
            self.set_completed(False)
            self.uncompleted_time_est += task.get_time_est()
        else:
            self.completed_time_est += task.get_time_est()
        self.update_total_time_est()
        self.value_est += task.get_prob() * task.get_reward()
        
    def set_completed(self, completed):
        self.completed = completed
        if completed:
            for task in self.tasks:
                task.set_completed(completed)
                
            # Change time-estimation values
            self.completed_time_est = self.total_time_est
            self.uncompleted_time_est = 0
            
    def reset_completed(self):
        self.completed = False
        self.uncompleted_time_est = 0
        self.completed_time_est = 0
        
        for task in self.tasks:
            task.set_completed(False)
            self.uncompleted_time_est += task.get_time_est()
            
        self.update_total_time_est()
            
    def update_total_time_est(self):
        self.total_time_est = self.completed_time_est + \
                              self.uncompleted_time_est


class Task:
    def __init__(self, description, time_est=1,
                 completed=False, goal=None, prob=1., reward=0):
        """
        
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
        self.time_est = time_est  # units of time required to perform a task
        
        self.completed = completed
        self.goal = goal
        self.prob = prob
        self.reward = reward
        
    def __str__(self):
        return f'Description: {self.description}\n' \
               f'Time est.: {self.time_est}\n' \
               f'Completed: {self.completed}\n' \
               f'Goal: {self.goal.get_description()}\n' \
               f'Probability: {self.prob}\n' \
               f'Reward: {self.reward}\n'
            
    def get_copy(self):
        return Task(self.description, self.time_est,
                    completed=self.completed, goal=self.goal,
                    prob=self.prob, reward=self.reward)
    
    def get_description(self):
        return self.description

    def get_goal(self):
        return self.goal

    def get_prob(self):
        return self.prob

    def get_reward(self):
        return self.reward
    
    def get_time_est(self):
        return self.time_est

    def is_complete(self):
        return self.completed

    def is_non_goal(self):
        return self.get_goal().is_non_goal()

    def set_goal(self, goal):
        self.goal = goal

    def set_completed(self, completed):
        if self.completed != completed:
            self.completed = completed
            if completed:
                self.goal.add_completed_time(self.time_est)
            else:
                self.goal.add_completed_time(-self.time_est)


class ToDoList:
    def __init__(self, goals, start_time=0, end_time=None, non_goal_tasks=None):
        """
        
        Args:
            goals: List of all goals
            start_time:
            end_time:
            non_goal_tasks: List of non-goal tasks
        """
        # Goals
        self.goals = goals
        self.completed_goals = set([goal for goal in self.goals
                                    if goal.is_complete()])
        self.incomplete_goals = set([goal for goal in self.goals
                                     if not goal.is_complete()])
        
        # Tasks
        self.tasks = []
        for goal in self.goals:
            self.tasks.extend(goal.get_tasks())
        self.completed_tasks = set([task for task in self.tasks
                                    if task.is_complete()])
        self.incomplete_tasks = set([task for task in self.tasks
                                     if not task.is_complete()])
        
        # Time
        self.time = start_time  # Current time
        self.start_time = start_time
        if end_time is None:
            max_deadline = float('-inf')
            for goal in self.goals:
                max_deadline = max(max_deadline, goal.get_deadline_time())
            end_time = max_deadline + 1
        self.end_time = end_time

        # Add non-goal tasks
        # self.non_goal_tasks = non_goal_tasks
        # if self.non_goal_tasks is None:
        #     self.non_goal_tasks = [
        #         Task("Non-goal Task", time_est=1, prob=1.0, reward=0)
        #     ]
        # self.non_goal = Goal(description="Non-goal", tasks=self.non_goal_tasks,
        #                      reward={float('inf'): 0}, non_goal=True)
        # self.goals += [self.non_goal]
        
    def action(self, task=None):
        """
        Do a specified action
        
        Args:
            task: ...; If not defined, it completes a random task

        Returns:

        """
        if task is None:
            task = random.choice(self.tasks)
        
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

        for goal in self.incomplete_goals:
            # Check:
            # 1) goal is now passed deadline at curr_time
            # 2) goal was not passed deadline at prev_time
            
            if curr_time > goal.get_deadline_time() and \
                    not prev_time > goal.get_deadline_time():
                penalty += goal.get_deadline_penalty()
                
        return penalty
    
    def do_task(self, task):
        goal = task.get_goal()
        threshold = task.get_prob()
        
        reward = task.get_reward()
        p = random.random()
        
        # check that task is completed on time
        # and NOT a non-goal and goal was not already complete before task
        if p < threshold and self.time <= goal.get_deadline_time() and \
                not task.is_non_goal() and not goal.is_complete():
                
            task.set_completed(True)
            self.incomplete_tasks.discard(task)
            self.completed_tasks.add(task)
            
            # If completion of the task completes the goal
            if goal.is_complete():
                reward += goal.get_reward(self.time)  # Goal completion reward
                self.incomplete_goals.discard(goal)
                self.completed_goals.add(goal)
                
        return reward

    def print_debug(self):
        """
        print ToDoList object

        Returns:

        """
        print("Current Time: " + str(self.time))
        print("Goals: " + str(self.goals))
        print("Completed Goals: " + str(self.completed_goals))
        print("Tasks: " + str(self.tasks))
        print("Completed Tasks: " + str(self.completed_tasks))

    # ===== Getters =====
    # @staticmethod
    # def is_goal_complete(goal):
    #     """
    #     Method for checking whether a goal is complete by checking
    #     if all tasks are complete
    #     """
    #     if goal.is_complete():
    #         return True
    #     for task in goal.tasks:
    #         if not task.is_complete():
    #             return False
    #     goal.set_completed(True)
    #     return True

    def get_end_time(self):
        return self.end_time

    def get_non_goal_val(self):
        return self.get_non_goal_task().get_reward()

    # def get_non_goal_task(self):
    #     """
    #     TODO: better way to do this would combine a list of tasks into one
    #           instead of hard coding first task
    #     """
    #     if len(self.non_goal_tasks) > 1:
    #         return self.non_goal_tasks[0]
    #     else:
    #         return self.non_goal_tasks

    def get_goals(self):
        return self.goals

    def get_tasks(self):
        return self.tasks

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
        self.tasks.extend(goal.get_tasks())
        
        if goal.is_complete():
            self.completed_goals.add(goal)
        else:
            self.incomplete_goals.add(goal)

    def add_task(self, goal, task):
        """
        Adds task to the specified goal
        """
        self.tasks.append(task)
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

        # Non-goal
        # self.non_goal_task = to_do_list.get_non_goal_task()
        # if len(self.non_goal_task) > 0:
        #     self.non_goal_val = self.non_goal_task.get_reward()
        # else:
        #     self.non_goal_val = None

        # Create mapping of indices to tasks represented as list
        self.index_to_task = to_do_list.get_tasks()
        # if len(self.non_goal_task) > 0:
        #     self.index_to_task.append(self.non_goal_task)  # Add non-goal task
        
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
        num_tasks = len(to_do_list.get_tasks())
        for t in range(self.to_do_list.get_end_time() + 2):
            for bit_vector in itertools.product([0, 1], repeat=num_tasks):
                state = (bit_vector, t)
                self.states.append(state)

        # Mapping from (binary vector x time) to integer (?!)
        self.state_to_index = {self.states[i]: i
                               for i in range(len(self.states))}

        self.reverse_DAG = MDPGraph(self)
        self.linearized_states = self.reverse_DAG.linearize()

        self.V_states = {}
        self.optimal_policy = {}
        
        # Perform backward induction
        # self.V_states, self.optimal_policy = \
        #     self.get_optimal_values_and_policy()

        # Pseudo-rewards
        self.pseudo_rewards = {}  # {(s, a, s') --> PR(s, a, s')}
        self.transformed_pseudo_rewards = {}  # {(s, a, s') --> PR'(s, a, s')}
        # self.calculate_pseudo_rewards()  # Calculate PRs for each state
        # self.transform_pseudo_rewards()  # Apply linear transformation to PR'

    def calculate_pseudo_rewards(self):
        """
        private method for calculating untransformed pseudo-rewards PR
        """
        for state in self.states:
            for action in self.get_possible_actions(state):
                for next_state, prob in \
                        self.get_trans_states_and_probs(state, action):
                    reward = self.get_reward(state, action, next_state)
                    pr = self.V_states[next_state][0] - \
                        self.V_states[state][0] + reward
                    self.pseudo_rewards[(state, action, next_state)] = pr

    @staticmethod
    def tasks_to_binary(tasks):
        """
        Convert a list of Task objects to a bit vector with 1 being complete and
        0 if not complete.
        """
        return tuple([1 if task.is_complete() else 0 for task in tasks])

    def transform_pseudo_rewards(self, print_values=False):
        """
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

        alpha = (highest + sec_highest) / 2
        beta = 1
        if alpha <= 1.0:
            beta = 10

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

    def get_optimal_values_and_policy(self):
        """
        Given a ToDoListMDP, perform value iteration/backward induction to find
        the optimal value function

        Input: ToDoListMDP
        Output: Dictionary of optimal value of each state
        """
        
        optimal_policy = {}  # state --> action
        v_states = {}  # state --> (value, action)

        # Perform Backward Iteration (Value Iteration 1 Time)
        for state in self.get_linearized_states():
            v_states[state] = self.get_value_and_action(state, v_states)
            optimal_policy[state] = v_states[state][1]
    
        return v_states, optimal_policy

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
        tasks = state[0]
        
        if not self.is_terminal(state):
            actions = [i for i, task in enumerate(tasks) if task == 0]
            # if len(self.non_goal_task) > 0:
            #     return actions + [-1]
            # else:
            return actions
        
        return []  # Terminal state --> No actions

    def get_q_value(self, state, action, v_states):
        """

        Args:
            state: current state (tasks, time)
            action: index of action in MDP's tasks
            v_states: dictionary mapping states to current best (value, action)

        Returns:
            Q-value of state
        """
        q_value = 0

        for next_state, prob in self.get_trans_states_and_probs(state, action):
            # tasks, time = next_state

            # IMPORTANT: Below varies on value iteration or policy iteration
            v = v_states[next_state]

            if isinstance(v, tuple):
                next_state_value = v_states[next_state][0]
            else:
                next_state_value = v_states[next_state]

            q_value += prob * (self.get_reward(state, action, next_state) +
                               self.get_gamma() * next_state_value)

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
    
        # Action is NOT a non-goal task
        if action != -1:
            # Reward for goal completion
            if next_tasks[action] == 1:
                if self.is_goal_completed(goal, next_state) and \
                        self.is_goal_active(goal, next_time):
                    reward += goal.get_reward(next_time)
    
        # Penalty for missing a deadline
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
        start_state = self.tasks_to_binary(self.to_do_list.get_tasks())
        return start_state, 0  # curr_state, self.get_time()

    def get_state_index(self, state):
        return self.state_to_index[state]

    def get_state_value(self, state):
        return self.V_states[state][0]
    
    def get_states(self):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        return self.states

    def get_tasks_list(self):
        return self.index_to_task

    def get_trans_states_and_probs(self, state, action):
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

        # Action is non-goal action
        if action is None:
            return next_states_probs  # Empty list / Terminal state
        
        if action == -1:
            binary_tasks, time = state
            time += 1
            next_states_probs.append(((tuple(binary_tasks), time), 1))
            return next_states_probs

        # Action is the index that is passed in
        task = self.index_to_task[action]
        binary_tasks = list(state[0])[:]
        new_time = state[1] + task.get_time_est()

        if new_time > self.to_do_list.get_end_time():
            new_time = self.to_do_list.get_end_time() + 1

        # State for not completing task
        tasks_no_completion = binary_tasks[:]
        if 1 - task.get_prob() > 0:  # 1 - P(completion)
            next_states_probs.append(((tuple(tasks_no_completion), new_time),
                                      1 - task.get_prob()))
            
        # State for completing task
        tasks_completion = binary_tasks[:]
        tasks_completion[action] = 1
        if task.get_prob() > 0:
            next_states_probs.append(((tuple(tasks_completion), new_time),
                                      task.get_prob()))

        return next_states_probs

    def get_value_and_action(self, state, v_states):
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
            q_value = self.get_q_value(state, action, v_states)
            if best_value < q_value:
                best_value = q_value
                best_action = action

        return best_value, best_action

    def get_value_function(self):
        """
        To get the state value for a given state, use get_state_value(state)
        
        Returns:

        """
        return self.V_states

    def is_goal_active(self, goal, time):
        """
        Given a Goal object and a time
        Check if the goal is still active at that time
        Note: completed goal is still considered active if time has not passed
              the deadline
        """
        return time <= goal.get_deadline_time() and \
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
        self.vertices = []

        # Initialize variables
        self.counter = None
        self.linearized_states = None
        
        # Connecting the graph in reverse manner | next_state --> curr_state
        for state in mdp.get_states():
            self.vertices.append(state)
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
