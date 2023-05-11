#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# grudat23 projekt Love Redin testkod
import numpy as np
import pandas as pd
from discrete_markov import *
from random import randint

def assert_sum_of_rows(df):
    # Check if the sum of each row in a matrix is approximately equal to one or zero
    # The sum of a row should be zero if the row state has not occurred, otherwise one.
    row_sums = df.sum(axis=1)
    assert np.all(np.logical_or(np.isclose(row_sums, 1.0), np.isclose(row_sums, 0.0))), "The sum of rows is not equal to one or zero."


def test_transition_matrix():
    # Test 1: Simple case with degree 1
    array = ['A', 'B', 'A', 'A', 'A', 'B', 'C']
    expected_result = pd.DataFrame({'A': [0.5, 0.5, 0],
                                    'B': [0.5, 0, 0.5],
                                    'C': [0, 0, 0]},
                                    index=['A', 'B', 'C']).T
    result = transition_matrix(array)
    assert result.equals(expected_result), "Test 1 failed"
    assert_sum_of_rows(result)
    
    # Test 2: Case with degree 2
    array = ['A', 'B', 'A', 'A', 'B', 'C']
    expected_result = pd.DataFrame({'A-A': [0, 1, 0],
                                    'A-B': [0.5, 0, 0.5],
                                    'A-C': [0, 0, 0],
                                    'B-A': [1, 0, 0],
                                    'B-B': [0, 0, 0],
                                    'B-C': [0, 0, 0],
                                    'C-A': [0, 0, 0],
                                    'C-B': [0, 0, 0],
                                    'C-C': [0, 0, 0]},
                                    index=['A', 'B', 'C']).T
    result = transition_matrix(array, degree=2)
    assert_sum_of_rows(result)
    assert result.equals(expected_result), "Test 2 failed"
    
    # Test random Markov Chains to see that nothing breaks
    for i in range(5):
        for degree in range(1, 5):
            array = [randint(1,3) for j in range(1000)]
            result = transition_matrix(array, degree)
            assert_sum_of_rows(result)
    
    print("All transition_matrix tests passed")


def test_simulate_markov_chain():
    # Test 1: Simple case with degree 1
    transition_df = pd.DataFrame({'A': [0, 1],
                                  'B': [1, 0]},
                                  index=['A', 'B'])
    initial_state = 'A'
    n_time_periods = 6
    expected_result = ['A', 'B', 'A', 'B', 'A', 'B']
    result = simulate_markov_chain(transition_df, initial_state, n_time_periods)
    assert result == expected_result, "Test 1 failed"
    
    # Test 2: Invalid initial state not in transition matrix columns
    invalid_initial_state = 'C'
    try:
        simulate_markov_chain(transition_df, invalid_initial_state, n_time_periods)
    except ValueError:
        pass
    else:
        assert False, "Test 2 failed. Expected ValueError."

    # Test 3: Non-integer n_time_periods
    non_integer_n_time_periods = 3.5
    try:
        simulate_markov_chain(transition_df, initial_state, non_integer_n_time_periods)
    except TypeError:
        pass
    else:
        assert False, "Test 3 failed. Expected TypeError."

    # Test 4: n_time_periods less than 1
    n_time_periods_less_than_one = 0
    try:
        simulate_markov_chain(transition_df, initial_state, n_time_periods_less_than_one)
    except ValueError:
        pass
    else:
        assert False, "Test 4 failed. Expected ValueError."

    print("All simulate_markov_chain tests passed")
    
    
def test_compute_stationary_distribution():
    # Test 1: Valid case
    transition_df = pd.DataFrame({'A': [0.2, 0.5, 0.3],
                                  'B': [0.4, 0.1, 0.5],
                                  'C': [0.1, 0.2, 0.7]},
                                  index=['A', 'B', 'C']).T
    expected_result = pd.DataFrame({'A': [0.188889, 0.233333, 0.577778],
                                    'B': [0.188889, 0.233333, 0.577778],
                                    'C': [0.188889, 0.233333, 0.577778]},
                                    index=['A', 'B', 'C']).T
    result = compute_stationary_distribution(transition_df)
    assert np.allclose(result, expected_result), "Test 1 failed. Expected:\n{}\nActual:\n{}".format(expected_result, result)

    # Test 2: Invalid case - Incorrect dimensions
    invalid_df = pd.DataFrame({'A': [0.2, 0.5, 0.3],
                               'B': [0.4, 0.1, 0.5]},
                               index=['A', 'B', 'C']).T
    try:
        compute_stationary_distribution(invalid_df)
    except ValueError:
        pass
    else:
        assert False, "Test 2 failed. Expected ValueError."

    print("All compute_stationary_distribution tests passed")
    
    
def main():
    """
    Runs tests for transition_matrix, simulate_markov_chain and
    compute_stationary_distribution from discrete_markov.
    
    """
    test_transition_matrix()
    test_simulate_markov_chain()
    test_compute_stationary_distribution()


if __name__ == '__main__':
    main()    