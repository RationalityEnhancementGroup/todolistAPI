"""
Tests for the algorithms that solve the to-do list MDP.
The test cases are provided in the todolistMDP/test_cases.py file.
"""

from todolistMDP.mdp_solvers import *
from todolistMDP.test_cases import *
from todolistMDP.to_do_list import *
from todolistMDP.scheduling_solvers import simple_goal_scheduler
from pprint import pprint


def follow_policy(mdp):
    """
    Prints the optimal policy.
    
    Args:
        mdp: ToDoListMDP.

    Returns:
        /
    """
    policy = mdp.get_optimal_policy()
    values = mdp.get_value_function()
    
    # Get initial state
    state = sorted(list(policy.keys()))[0]
    vec, tm = state

    action = policy[state]
    value = values[state]

    print(state, action, value)
    
    while policy[state] is not None:
        # Get task
        idx = policy[state]
        task = mdp.index_to_task[idx]
        
        # Get next state
        vec = list(vec)
        vec[idx] = 1
        
        tm += task.get_time_est()
        state = (tuple(vec), tm)
        
        action = policy[state]
        value = values[state]

        print(state, action, value)

    return


def solve(to_do_list, solver, print_policy=False, print_pseudo_rewards=False,
          print_values=False):
    """
    Runs one of the algorithms from the old report, based on MDP formulation.
    
    Args:
        to_do_list: ToDoList
        solver: Solving algorithm. Valid values:
            - backward_induction
            - policy_iteration
            - value iteration
        print_policy: Whether to print the optimal/output policy or not.
        print_pseudo_rewards: Whether to print the pseudo-rewards or not.
        print_values: Whether to print the value function or not.

    Returns:
        Solved MDP.
    """
    
    start = time.time()
    print(f'===== {solver.__name__} =====')
    mdp = solver(to_do_list)
    print(f'{solver.__name__} takes {time.time() - start:.4f} seconds.\n')
    
    if print_policy:
        print("Optimal Policy:")
        pprint(mdp.get_optimal_policy())
        print('\n')

    if print_pseudo_rewards:
        print("Pseudo-rewards:")
        pprint(sorted(mdp.get_pseudo_rewards().values())[::-1])
        pprint(mdp.get_pseudo_rewards())
        print('\n')

    if print_values:
        print("Value Function:")
        pprint(mdp.get_value_function())
        print('\n')

    follow_policy(mdp)

    return mdp


# Set of goals to use
goals = d_different_value_extra_time_deadlines

# Generate to-do list MDP
s_time = time.time()  # Start timer
to_do_list = ToDoList(goals, start_time=0)

# ===== Backward induction =====
"""
    (+) Deterministic case (6 goals x 2 tasks)
    (+) Simple deterministic case (2 goals x 2 tasks)
    (+) Probabilistic case (2 goals x 2 tasks)
"""
# mdp = ToDoListMDP(to_do_list)
# print(f'MDP initialization takes {time.time() - s_time:.4f} seconds.')
# mdp = solve(to_do_list, backward_induction)

# ===== Policy iteration =====
"""
    (?) Deterministic case (6 goals x 2 tasks)
      - Fills up the memory and the computer freezes...
    (+) Simple deterministic case (2 goals x 2 tasks)
    (+) Probabilistic case (2 goals x 2 tasks)
"""
# mdp = ToDoListMDP(to_do_list)
# print(f'MDP initialization takes {time.time() - s_time:.4f} seconds.')
# mdp = solve(to_do_list, policy_iteration)

# ===== Value iteration =====
"""
    (+) Deterministic case (6 goals x 2 tasks)
    (+) Simple deterministic case (2 goals x 2 tasks)
    (+) Probabilistic case (2 goals x 2 tasks)
"""
# mdp = ToDoListMDP(to_do_list)
# print(f'MDP initialization takes {time.time() - s_time:.4f} seconds.')
# mdp = solve(to_do_list, value_iteration)

# ===== Simple scheduler =====
start_time = time.time()
tasks_list = simple_goal_scheduler(to_do_list, mixing_parameter=0.90,
                                   verbose=True)

current_time = 0
for task in tasks_list:
    print(f'Current time: {current_time}')
    print(task)
    current_time += task.get_time_est()
    
print(f'Scheduling goals with DP algorithm took '
      f'{int(time.time() - start_time)} seconds!')
