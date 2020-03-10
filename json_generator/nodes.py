import json
import random
import time

HEX_DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              'a', 'b', 'c', 'd', 'e', 'f']
NODE_COUNTER = 0


class Node:
    def __init__(self, id, deadline=None, parent=None, points=None,
                 prob_success=None, time_est=None):
        """
        Initializes a new node with the provided parameters.
        
        Args:
            id: ID of the node.
            deadline: Deadline of the node.
            parent: Pointer to the parent node.
            points: Number of points of the node.
            prob_success: Probability of success of the node.
            time_est: Time estimation of the node.
        """
        
        # Set parameters
        if id is None:
            self.id = self.get_random_id()
        elif type(id) is int:
            self.id = id
            # self.id = self.int_to_id(id)  # Un-comment this for HEX code
        else:
            self.id = id

        self.deadline = deadline  # Deadline
        self.lm = self.get_current_time()  # Last modified
        self.parent = parent  # Parent
        self.points = points  # Points
        self.prob_success = prob_success  # Probability of success
        self.time_est = time_est  # Time estimation

        self.ch = []  # Children nodes
        # Set depth
        if self.parent is None:  # Root node
            self.depth = 0
        else:  # Internal/Leaf node
            self.depth = self.parent.depth + 1
            
        # Generate name
        self.nm = self.generate_nm()

    def __len__(self):
        """
        Returns:
            Length of the list of children nodes.
        """
        return len(self.ch)
    
    def __str__(self):
        return f'Id: {self.id}\n' \
               f'Name: {self.nm}\n' \
               f'Deadline: {self.deadline}\n' \
               f'Depth: {self.depth}\n' \
               f'Last modified: {self.lm}\n' \
               f'Parent ID: ' \
               f'{self.parent.id if self.parent is not None else None}\n' \
               f'Points: {self.points}\n' \
               f'Time estimation: {self.time_est}\n'
    
    @staticmethod
    def get_current_time():
        return int(round(time.time() * 1000))
    
    @staticmethod
    def get_random_id():
        """
        Generates a random hexadecimal ID in the format
        00000000-0000-0000-0000-{12x:hex_number}
        
        Returns:
            Hexadecimal ID in the format
            00000000-0000-0000-0000-{12x:hex_number}
        """
        return '-'.join([
            random.choices(HEX_DIGITS, k=8),
            random.choices(HEX_DIGITS, k=4),
            random.choices(HEX_DIGITS, k=4),
            random.choices(HEX_DIGITS, k=4),
            random.choices(HEX_DIGITS, k=12)
        ])
    
    @staticmethod
    def int_to_id(int_number):
        """
        Converts a provided integer number into an ID code in the format
        00000000-0000-0000-0000-{12x:hex_number}

        Args:
            int_number: Non-negative integer number.

        Returns:
            ID in the format 00000000-0000-0000-0000-{12x:hex_number}
        """
        # Convert the integer number into a hexadecimal number
        hex_number = hex(int_number)
        hex_number = str(hex_number)[2:]  # Remove '0x'
        
        return '-'.join([
            '0' * 8,
            '0' * 4,
            '0' * 4,
            '0' * 4,
            '0' * (12 - len(hex_number)) + hex_number
        ])

    def get_ch(self):
        return self.ch
    
    def get_deadline(self):
        return self.deadline
    
    def get_depth(self):
        return self.depth
    
    def get_id(self):
        return self.id
    
    def get_lm(self):
        return self.lm
    
    def get_nm(self):
        return self.nm

    def get_parent(self):
        return self.parent

    def get_prob_success(self):
        return self.prob_success

    def get_points(self):
        return self.points
    
    def get_time_est(self):
        return self.time_est

    def generate_tree(self, ignore_root=True):
        """
        Generates a JSON tree structure starting from this node.
        
        Args:
            ignore_root: Whether to ignore or include info for this node.

        Returns:
            JSON-formatted tree of nodes.
        """
        self.update_time_est()  # Update time-estimation values of the tree
        
        def get_dict(node):
            """
            Makes recursive calls in order to get info for this node and its
            children nodes.
            
            Args:
                node: Node whose info is obtained.

            Returns:
                Information for children nodes.
            """
            return {
                "id": node.get_id(),
                "nm": node.get_nm(),
                "lm": node.get_lm(),
                "ch": [
                    get_dict(ch) for ch in node.get_ch()
                ]
            }
        
        # If the info for this node should be included
        if ignore_root:
            return [get_dict(ch) for ch in self.get_ch()]

        # If the info for this node should be excluded
        return get_dict(self)

    def generate_nm(self):
        """
        Generates the name of this node.
        
        Returns:
            The newly-generated name.
        """
        # If it is a root node
        if self.depth == 0:
            self.nm = f'ROOT_NODE'
        
        # If it is a goal node
        if self.depth == 1:
            self.nm = f'#CG{self.id}'
            
        # If it is a task node
        if self.depth >= 2:
            self.nm = f'#T{self.id}'

        # Append other information to the name of the node
        if self.points is not None:
            self.nm += f' =={self.points}'
        if self.time_est is not None:
            self.nm += f' ~~{self.time_est}min'
        if self.deadline is not None:
            self.nm += f' DUE: {self.deadline}'

        return self.nm

    def generate_nodes(self, deadlines=None, points=None, prob_success=None,
                       time_est=None, deadline_val=None, point_val=None,
                       prob_success_val=None, time_est_val=None, num_nodes=1):
        """
        Generate new nodes with the provided parameters as children of this
        node. It is mandatory to provide list of points or list of time
        estimations.

        Args:
            deadlines: List of deadlines.
            points: List of points.
            prob_success: List of probabilities of successful completion.
            time_est: List of time estimations.
            deadline_val: Value to fill unprovided deadlines.
            point_val: Value to fill unprovided points.
            prob_success_val: Value to fill unprovided probabilities.
            time_est_val: Value to fill unprovided time estimations.

        Returns:
            List of newly-generated nodes.
        """
        global NODE_COUNTER
        
        if points is None:
            points = [point_val for _ in range(num_nodes)]
        if time_est is None:
            time_est = [time_est_val for _ in range(num_nodes)]

        num_nodes = len(time_est)
        
        # Generate missing data
        if deadlines is None:
            deadlines = [deadline_val for _ in range(num_nodes)]
        if prob_success is None:
            prob_success = [prob_success_val for _ in range(num_nodes)]

        # Check whether all lists have the same length
        assert len(deadlines) == len(points) == len(prob_success) == \
               len(time_est) == num_nodes

        # Generate new nodes
        nodes = [
            Node(id=NODE_COUNTER + idx + 1, deadline=deadlines[idx],
                 parent=self, points=points[idx],
                 prob_success=prob_success[idx], time_est=time_est[idx])
            for idx in range(num_nodes)
        ]
        self.ch += nodes  # Append new nodes as children to the parent node
        NODE_COUNTER += num_nodes  # Update node counter
            
        return nodes
    
    def print_ch(self):
        """
        Print info for the children nodes of this node.
        
        Returns:
            /
        """
        print('\n'.join(ch.__str__() for ch in self.ch))
        return
    
    def set_ch_deadlines(self, deadlines):
        # Check whether the length of both children nodes and deadline matches
        if len(self) != len(deadlines):
            print(f'ERROR: The number of input deadlines does not correspond '
                  f'to the number of children nodes of node with ID {self.id}.')
        assert len(deadlines) == len(self)
        
        # Set new deadline for each of the children nodes
        for idx in range(len(self)):
            self.ch[idx].set_deadline(deadlines[idx])
            
        # Update the time of last modification
        self.lm = self.get_current_time()
        
        # Update the name of the node
        self.generate_nm()
        
        return

    def set_deadline(self, deadline):
        """
        Set new deadline to this node. Format YYYY-MM-DD
        
        Args:
            deadline: New deadline.

        Returns:
            /
        """
        # Set new deadline
        self.deadline = deadline
        
        # Update the time of last modification
        self.lm = self.get_current_time()
        
        # Update the name of the node
        self.generate_nm()  # Update name
        return

    def set_points(self, points):
        """
        Set new points value to this node.
        
        Args:
            points: New points value.

        Returns:
            /
        """
        self.points = points
        self.lm = self.get_current_time()
        self.generate_nm()  # Update name
        return

    def set_time_est(self, time_est):
        """
        If this node is a leaf node, then it sets the new time-estimation value.
        If this node is not a leaf node, then it broadcasts the provided
        time-estimation value to the leaf nodes of this node.
        
        Args:
            time_est: Value of the new time estimation.

        Returns:
            /
        """
        # If this node is a leaf node (no children)
        if len(self) == 0:
            self.time_est = time_est
            self.update_last_modified()
            self.generate_nm()  # Update name
            
        # If this node is not a leaf node, then make a recursive call from each
        # children node in order to propagate the value to the leaf nodes.
        else:
            self.time_est = 0
            for ch in self.ch:
                ch.set_time_est(time_est)
                self.time_est += ch.get_time_est()

        return
                
    def update_last_modified(self):
        """
        Sets last-modified timestamp to the moment of the latest update.
        
        Returns:
            /
        """
        self.lm = self.get_current_time()
        return
        
    def update_time_est(self):
        """
        Updates time estimation of the given (sub-)tree.
        
        Returns:
            /
        """
        # If this node is a leaf node
        if len(self) != 0:
            
            # Make recursive calls in order to update time estimation
            self.time_est = 0
            for ch in self.ch:
                self.time_est += ch.update_time_est()
                
            self.update_last_modified()  # Change last modified timestamp
            self.generate_nm()  # Update name

        return self.get_time_est()
