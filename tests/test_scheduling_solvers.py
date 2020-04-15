import pytest

from todolistMDP.scheduling_solvers import *
from todolistMDP.test_cases import *


def test_greedy_method_and_mixing_time():
    """
    Test cases for the greedy method in todolistMDP/scheduling_solvers.py
    """
    
    """
    Negative rewards, all attainable goals.
    """
    goals = deterministic_tests["negative_reward_attainable_goals"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    result = get_attainable_goals_greedy(goals)
    assert len(result) == len(goals)
    assert sum(goal.get_reward(0) for goal in result) == -1111
    
    """
    1 unattainable high-reward goal, 3 attainable goals.
    """
    goals = deterministic_tests["unattainable_high_reward_goal"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    with pytest.raises(Exception):
        assert get_attainable_goals_greedy(goals)
    
    """
    1 unattainable low-reward goal, 3 attainable goals.
    """
    goals = deterministic_tests["unattainable_low_reward_goal"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    with pytest.raises(Exception):
        assert get_attainable_goals_greedy(goals)
    
    """
    All different deadlines with additional scheduling time (non-sharp).
    """
    goals = deterministic_tests["all_different_extra_time_deadlines"]
    goals.sort()
    result = get_attainable_goals_greedy(goals)
    
    # Test get_attainable_goals_greedy
    assert len(result) == len(goals)
    assert sum(goal.get_reward(0) for goal in result) == 4
    
    # Test compute_mixing_time
    assert (compute_mixing_time(result) == np.array([5, 5, 5, 5])).all()
    
    """
    Latest deadline: 10 years (in minutes with 24 hour workload / day)
    """
    goals = deterministic_tests["distant_deadlines"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    result = get_attainable_goals_greedy(goals)
    assert len(result) == len(goals)
    assert sum(goal.get_reward(0) for goal in result) == 10
    
    # Test compute_mixing_time function
    mixing_time = compute_mixing_time(goals)
    arr = np.array([unit * YEAR_TO_MINS - unit * 10 for unit in range(1, 11)])
    assert (mixing_time == arr).all()
    
    """
    one_mixing
    - The actual mixing is not tested because the ouptut is not deterministic!
    """
    goals = deterministic_tests["one_mixing"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    result = get_attainable_goals_greedy(goals)
    assert len(result) == len(goals)
    assert sum(goal.get_reward(0) for goal in result) == 4
    
    # Test compute_mixing_time function
    mixing_time = compute_mixing_time(goals)
    assert (mixing_time == np.array([0, 5, 0, 0])).all()
    
    """
    All deadlines are in the past.
    """
    goals = deterministic_tests["negative_value_deadlines"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    with pytest.raises(Exception):
        assert get_attainable_goals_greedy(goals)
    
    """
    Some deadlines are in the past.
    """
    goals = deterministic_tests["partially_negative_value_deadlines"]
    goals.sort()
    with pytest.raises(Exception):
        assert get_attainable_goals_greedy(goals)
    
    """
    All deadlines are at the same time with additional time (non-sharp).
    """
    goals = deterministic_tests["same_value_extra_time_deadlines"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    result = get_attainable_goals_greedy(goals)
    assert len(result) == len(goals)
    assert sum(goal.get_reward(0) for goal in result) == 4
    
    # Test compute_mixing_time function
    mixing_time = compute_mixing_time(goals)
    assert (mixing_time == np.array([40, 30, 20, 10])).all()
    
    """
    All deadlines are at the same time. The last deadline is sharp.
    """
    goals = deterministic_tests["same_value_sharp_deadlines"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    result = get_attainable_goals_greedy(goals)
    assert len(result) == len(goals)
    assert sum(goal.get_reward(0) for goal in result) == 4
    
    # Test compute_mixing_time function
    mixing_time = compute_mixing_time(goals)
    assert (mixing_time == np.array([30, 20, 10, 0])).all()
    
    """
    No deadline is attainable.
    """
    goals = deterministic_tests["same_value_unattainable_deadlines"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    with pytest.raises(Exception):
        assert get_attainable_goals_greedy(goals)
    
    """
    All deadlines are attainable exactly when they finish (sharp).
    """
    goals = deterministic_tests["sharp_deadlines"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    result = get_attainable_goals_greedy(goals)
    assert len(result) == len(goals)
    assert sum(goal.get_reward(0) for goal in result) == 4
    
    # Test compute_mixing_time function
    mixing_time = compute_mixing_time(goals)
    assert (mixing_time == np.array([0, 0, 0, 0])).all()
    
    """
    All deadlines are at the current time (0 minutes to the deadline).
    """
    goals = deterministic_tests["zero_value_deadlines"]
    goals.sort()
    
    # Test get_attainable_goals_greedy
    with pytest.raises(Exception):
        assert get_attainable_goals_greedy(goals)

# TODO: Find a way to test:
#       - compute_mixing_values
#       - get_ordered_task_list
#       - shuffle

# TODO: test_dp_method():
#       - compute_gcd
#       - compute_optimal_values
#       - get_attainable_goals_dp
#       - scale_time
