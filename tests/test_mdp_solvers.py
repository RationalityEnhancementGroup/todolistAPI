"""
Tests for the algorithms that solve the to-do list MDP.
The test cases are provided in the todolistMDP/test_cases.py file.
"""

import pytest
from todolistMDP.mdp_solvers import *
from todolistMDP.test_cases import *
from todolistMDP.to_do_list import *
from todolistMDP.scheduling_solvers \
    import run_algorithm, run_dp_algorithm, run_greedy_algorithm
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
    
    visited_states = deque([state])
    
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
        
        visited_states.append(state)

        print(state, action, value)

    return visited_states


def print_pseudo_rewards(mdp, states):
    for idx in range(len(states) - 1):
        s = states[idx]
        a = mdp.get_optimal_policy(s)
        s_ = states[idx + 1]
        print(mdp.pseudo_rewards[(s, a, s_)])


def print_transformed_pseudo_rewards(mdp, states):
    for idx in range(len(states) - 1):
        s = states[idx]
        a = mdp.get_optimal_policy(s)
        s_ = states[idx + 1]
        print(mdp.transformed_pseudo_rewards[(s, a, s_)])


def solve(to_do_list, solver,
          print_policy=False, print_pseudo_rewards=False, print_values=False):
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


def test_multiple_test_cases(test_cases, algorithm,
                             mixing_parameter=0.0, verbose=False):
    for test_type in test_cases.keys():
        
        # Set of goals to use
        goals = test_cases[test_type]
        
        # Generate to-do list MDP
        to_do_list = ToDoList(goals, start_time=0)
        
        start_time = time.time()
        
        print(f"{test_type}...")
        
        try:
            if algorithm == run_dp_algorithm or \
                    algorithm == run_greedy_algorithm:
                run_algorithm(to_do_list,
                              algorithm_fn=algorithm,
                              mixing_parameter=mixing_parameter,
                              verbose=verbose)
            else:
                print_policy = False or verbose
                print_pseudo_rewards = False or verbose
                print_values = False or verbose
                
                solve(to_do_list, algorithm,
                      print_policy=print_policy,
                      print_pseudo_rewards=print_pseudo_rewards,
                      print_values=print_values)
            
            print(f'Running the algorithm took '
                  f'{time.time() - start_time:.4f} seconds!')
        except Exception as error:
            print(str(error))
        
        print()
        
    return None


# Set of goals to use
# goals = generate_deterministic_test(num_goals=1000, num_tasks=10)
goals = d_bm

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
print(f'MDP initialization takes {time.time() - s_time:.4f} seconds.')
mdp = solve(to_do_list, backward_induction)
# pprint(mdp.v_states)
# print(mdp.pseudo_rewards)
# print(mdp.transformed_pseudo_rewards)

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

# ===== DP / Greedy algorithm =====
# start_time = time.time()
#
# tasks_list = run_algorithm(to_do_list,
#                            algorithm_fn=run_greedy_algorithm,
#                            mixing_parameter=0.0, verbose=False)
# print(f'Running the algorithm took '
#       f'{time.time() - start_time:.4f} seconds!')

# test_multiple_test_cases(deterministic_tests, run_greedy_algorithm, verbose=True)

# current_time = 0
# for task in tasks_list:
#     print(f'Current time: {current_time}')
#     print(task)
#     current_time += task.get_time_est()

# for num_goals in [1000]:
#     for mixing_parameter in [0.0, 0.1, 0.25, 0.5, 0.75, 0.90, 0.99]:
#         print(f"Goals: {num_goals} | "
#               f"Mixing parameter {mixing_parameter}")
#
#         goals = generate_deterministic_test(num_goals=num_goals, num_tasks=1)
#         to_do_list = ToDoList(goals, start_time=0)
#
#         run_algorithm(to_do_list, algorithm_fn=run_greedy_algorithm,
#                       mixing_parameter=mixing_parameter, verbose=False)
#
#         print()

# ===== Comparison ====
# for num_tasks in range(1, 5):
#     print(f"Num tasks: {4 * num_tasks}")
#     start_time = time.time()
#     goals = generate_deterministic_test(num_goals=4, num_tasks=num_tasks)
#     to_do_list = ToDoList(goals, start_time=0)
#     run_algorithm(to_do_list, algorithm_fn=run_greedy_algorithm,
#                   mixing_parameter=0, verbose=False)
#     print(f'Running the algorithm took '
#           f'{time.time() - start_time:.4f} seconds!')
