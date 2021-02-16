import numpy as np
import time
from toolz import memoize, compose
from collections import deque
from copy import deepcopy
from todolistMDP.zero_trunc_poisson import get_binned_dist
from pprint import pprint
import itertools


class Item:

    def __init__(self, description, completed=False, deadline=None,
                 deadline_datetime=None, item_id=None, children=None, value=None,
                 parent_item=None, essential=False, importance=0, time_est=0, today=None,
                 intrinsic_reward=0):
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

        # Add items on the next level/depth
        if children is not None:
            self.add_children(children)
            self.time_est = sum([child.time_est for child in self.children])

        self.optimal_reward = 0

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
                self.done_children.add(child)

            # Add child
            self.children.append(child)

            # Set goal as a parent item
            child.set_parent_item(self)

        # Convert start state from list to tuple
        self.start_state = tuple(self.start_state)

        # Set number of children_
        self.num_children = len(self.start_state)

    def set_idx(self, idx):
        self.idx = idx

    def is_completed(self):
        return self.completed

    def set_parent_item(self, parent_item):
        self.parent_item = parent_item

    def is_today(self):
        return self.today

    def get_optimal_reward(self):
        return self.optimal_reward

    def solve(self, verbose=False):
        print("Solving for following items")
        for item in self.children:
            print(item.description)


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

        # Add root node
        self.root = Item(description="Root", completed=False, deadline=np.PINF, children=self.complete_to_do_list,
                         essential=True, value=0)

        for goal in self.root.children:
            self.set_parent(goal, self.root)

        # Get tasks
        self.tasks = tuple(self.flatten(self.get_tasks(self.root)))
        self.num_tasks = len(self.tasks)
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
        self.nodes = list(self.flatten(self.get_nodes(self.root, False)))
        self.nodes.insert(0, self.root)
        self.nodes = tuple(self.nodes)
        self.node_names = list(self.flatten(self.get_nodes(self.root, True)))
        self.node_names.insert(0, self.root.description)
        self.node_names = tuple(self.node_names)

        # Set Value dictionary used to store optimal rewards
        value_dict = dict()
        for node in tuple(self.node_names):
            value_dict[node] = 0
        self.value_dict = value_dict

        self.tree = dict()
        self.tree_recurse([self.root])

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

    def get_tasks(self, root):
        all_tasks = []
        for goal in root.children:
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
        while len(task_list) is not 0:
            task = task_list[0]
            print(f'Desc: {task.parent_item.description}, Rew: {task.parent_item.intrinsic_reward}')
            next_list.append(task.parent_item)
            in_sum = 0
            for child in task.parent_item.children:
                print(f'Child: {child.description}, {child.intrinsic_reward}')
                in_sum += child.intrinsic_reward
                task_list.remove(child)
            task.parent_item.intrinsic_reward = in_sum
            print(f'Updated rew: {task.parent_item.intrinsic_reward}')

        self.current_pass(next_list)

    def solve(self, start_time=None, verbose=False):
        print(f'*********** In MainToDoList solve ***********')
        if start_time is None:
            start_time = self.start_time
        params = {
            "loss_rate": self.loss_rate,
            "penalty_rate": self.penalty_rate
        }
        # Only goal value is treated as intrinsic reward. Value of completing root node = 0
        # Solve the small MDPs and go down the tree
        for node in self.tree:
            self.solve_MDP(self.tree[node], verbose)

    def solve_MDP(self, children_nodes, verbose):
        if len(children_nodes) == 0:  # No children, reached leaf node
            return
        print(f'Solving treating goal as: {children_nodes[0].parent_item.description}')
        tasks = deepcopy(children_nodes)
        for task in tasks:
            if children_nodes[0].parent_item.description == "Root":
                task.intrinsic_reward = task.value
            task.value = self.value_dict[task.description]
        #             print(f'{task.description}, {task.value}, {task.intrinsic_reward}')

        goal = Item(description="MiniMDP", completed=False, deadline=np.PINF, children=tasks, essential=True, value=0)

        goal.solve(verbose)
        # ================================= Computing Psuedo-rewards =================================
        prs = self.compute_start_state_pseudo_rewards(
            goal)
        incentivized_tasks = prs["incentivized_items"]
        # Convert task queue to a list
        optimal_tasks = list(incentivized_tasks)
        # Sort task in decreasing order w.r.t. optimal reward
        optimal_tasks.sort(key=lambda task: -task.get_optimal_reward())
        print("Optimal Rewards")
        for task in optimal_tasks:
            print(task.description, task.get_optimal_reward())
            self.value_dict[task.description] = task.get_optimal_reward()
        return

    def compute_start_state_pseudo_rewards(self, goal, bias=None, scale=None):
        # TO DO
        incentivized_items = goal.children
        id2pr = None
        sc_sum_pr = None
        scale = None
        bias = None
        return {
            "incentivized_items": incentivized_items,

            "id2pr": id2pr,
            "sc_sum_pr": sc_sum_pr,
            "scale": scale,
            "bias": bias
        }