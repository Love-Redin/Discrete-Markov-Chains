#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# grudat23 projekt Love Redin
"""
Package discrete_markov v1.0 provides functions for working with discrete Markov chains,
including calculating transition matrices, simulating Markov chains, and computing
stationary distributions.


Transition Matrix:
The transition_matrix function computes the transition matrix from a given array of
states, and a given degree. The resulting transition matrix is a pandas DataFrame, 
where each row represents the current state and each column represents the next state.
The values in the matrix represent the probabilities of transitioning from the current
state to the next state, where rows represent previous/current state and columns are 
next states.

The degree indicates how many previous states to consider, default is 1.


Simulate Markov Chain:
The simulate_markov_chain function simulates the evolution of a Markov chain over a
specified time period, given the transition matrix and an initial state. It returns
an array of states representing the simulated trajectory of the Markov chain.


Compute Stationary Distribution:
The compute_stationary_distribution function computes the stationary distribution
matrix of a Markov chain, given a transition matrix of degree one. The stationary
distribution represents the long-term probabilities of being in each state of the
Markov chain. Note that convergence may not occur for some Markov chains, such as
absorbing, periodic, or non-ergodic chains.


Visualize Transition Matrix
The visualize_transition_matrix function visualizes the Markov transition matrix 
using a horizontal bar chart. It takes a Pandas DataFrame representing the transition
matrix as input, where each column and row label represents a state. The values in 
the matrix represent the probabilities of transitioning from the current state to the
next state. The resulting plot provides an intuitive visualization of the transition
probabilities between states.


Limitations:
- The transition_matrix function is recommended for use for degrees up to approximately five,
as the number of rows and time complexity increase exponentially with higher degrees. 
- The compute_stationary_distribution function requires a square transition matrix (n x n),
and the rows and columns must have consistent naming. 
- The visualize_transition_matrix function is suitable for Markov transition matrices with
a reasonable number of states. For large matrices, the plot may become crowded and less 
interpretable.


It is recommended to carefully review the documentation and ensure proper usage of
the functions in this package for accurate and reliable results in working with 
discrete Markov chains.
"""
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt


def transition_matrix(array, degree=1):
    """
    Computes the transition matrix from a given array of states.

    Parameters:
    - array (list or numpy array): Array of states representing a sequence of observations.
    - degree (int, optional): Degree of the Markov chain, i.e., the number of previous states to consider. Default is 1.

    Returns:
    - transition_df (pandas DataFrame): Transition matrix as a DataFrame, where each row represents the current state
      and each column represents the next state. The values in the matrix represent the probabilities of transitioning
      from the current state to the next state. The sum of all rows is one.
    
    If n is the number of unique individual states in the array, and k is the degree, 
    the resulting transition matrix will contain n columns and n^k rows.
    
    It is therefore not recommended to use degree higher than approximately five,
    depending on how many unique states are possible, as the number of rows 
    (and time complexity) increases exponentially with degree.
    """
    
    unique_states = list(set(array))
    values = dict()
    
    for index in range(degree-1, len(array)-degree+1):
        current_state = "-".join([str(state) for state in array[index-degree+1:index+1]])
        try:
            next_state = array[index+1]
            if current_state in values:
                state_dict = values[current_state]
                if next_state in state_dict:
                    state_dict[next_state] += 1
                else:
                    state_dict[next_state] = 1
            else:
                values[current_state] = {next_state:1}
        except IndexError:
            break
    
    for current_state, state_dict in values.items():
        sum_values = sum(state_dict.values())
        for next_state, value in state_dict.items():
            state_dict[next_state] /= sum_values
        
    possible_previous_states = _get_possible_previous_states(unique_states, degree)
    possible_next_states = unique_states

    transition_df = pd.DataFrame(0, index=possible_previous_states, columns=possible_next_states)
    for previous_state in possible_previous_states:
        for next_state in possible_next_states:
            if previous_state in values and next_state in values[previous_state]:
                transition_df.loc[previous_state, next_state] = values[previous_state][next_state]

    transition_df = transition_df.sort_index(axis=0).sort_index(axis=1).astype(float)
    return transition_df


def _get_possible_previous_states(unique_states, degree):
    """
    Helper function for transition_matrix.
    Generates all possible previous states for a given array of unique individual states and degree.
    Return a list of possible previous states.
    -----------------------------------------
    Example: unique_states = ["A", "B", "C"] and degree = 2. This returns
    previous_states = ['A-A', 'A-B', 'A-C', 'B-A', 'B-B', 'B-C', 'C-A', 'C-B', 'C-C']
    """
    previous_states = []
    combinations = list(itertools.product(unique_states, repeat=degree))    
    previous_states = ['-'.join(map(str, combination)) for combination in combinations]
    
    return previous_states


def simulate_markov_chain(transition_matrix, initial_state, n_time_periods): 
    """
    Simulates the evolution of a degree one Markov Chain over a specified 
    time period, given the transition matrix of degree one and an initial state. 
    Returns an array of states.
    """
    # Check that initial state is in transition matrix columns
    if not (initial_state in transition_matrix.columns or str(initial_state) in transition_matrix.columns):
        raise ValueError(f"Invalid initial state '{initial_state}': not in transition matrix.")

    if not type(n_time_periods) == int or type(n_time_periods) == float and n_time_periods.is_integer():
        raise TypeError("Discrete Markov Chain: n_time_periods must be an integer.")
    
    if n_time_periods < 1:
        raise ValueError("Too short simulation: n_time_periods must be at least one.")
                
    current_state = initial_state
    states = [current_state]

    for t in range(1, n_time_periods):
        try:
            proba_vector = transition_matrix.loc[current_state]
        except KeyError:
            proba_vector = transition_matrix.loc[str(current_state)]
        current_state = np.random.choice(proba_vector.index, p=proba_vector.values)
        states.append(current_state)

    return states
    

def compute_stationary_distribution(transition_df):
    """
    Computes and returns the stationary distribution matrix
    of a Markov Chain, given a transition matrix of degree one.
    Note that some Markov chains may not converge, 
    e.g. Absorbing Markov Chains, Periodic Markov Chains
    and Non-ergodic Markov Chains.
    """
    terminal_matrix = transition_df.copy()
    
    if len(terminal_matrix.columns) != len(terminal_matrix.index):
        raise ValueError("Stationary distribution only supports n x n transition matrices.")
    
    def normalize(row):
        # Normalizes rows, so matrix elements 
        # do not diverge due to rounding errors.
        return row / np.sum(row)
    
    # Assert consistent naming of columns and rows, allowing us to perform matrix multiplication
    terminal_matrix.columns = sorted([str(state) for state in terminal_matrix.columns])
    terminal_matrix.index = sorted([str(state) for state in terminal_matrix.index])
    
    if not set(terminal_matrix.columns) == set(terminal_matrix.index):
        raise ValueError("Rows and columns must be the same.")

    for i in range(100):
        terminal_matrix = terminal_matrix.dot(terminal_matrix)
        terminal_matrix = terminal_matrix.apply(normalize, axis=1)
            
    return terminal_matrix        


def visualize_transition_matrix(transition_df):
    """
    Visualizes the Markov transition matrix using a horizontal bar chart.

    Parameters:
    - transition_df (pandas DataFrame): DataFrame representing the Markov transition matrix,
      where each column and row label represents a state.

    Returns:
    - fig, ax (matplotlib Figure and Axes): Figure and Axes objects of the generated plot.

    The transition_df should be a transition matrix where each row and column corresponds to a state.
    The values in the matrix represent the probabilities of transitioning from the current state to 
    the next state.
    The sum of probabilities in each row should be one.

    Note: This visualization is suitable for Markov transition matrices with a reasonable number of states.
    For large matrices, the plot may become crowded and less interpretable.
    """
    previous_states = transition_df.index
    current_states = transition_df.columns
    if not (len(previous_states) >= len(current_states) > 0):
        raise ValueError("Incorrect shape of transition matrix.")
    transition_data = np.array(transition_df.values)
    data_cumulative = transition_data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn_r')(np.linspace(0.15, 0.85, transition_data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(transition_data, axis=1).max())

    for i, (colname, color) in enumerate(zip(current_states, category_colors)):
        transition_widths = transition_data[:, i]
        transition_starts = data_cumulative[:, i] - transition_widths
        ax.barh(previous_states, transition_widths, left=transition_starts, height=0.9,
                label=colname, color=color)

    ax.legend(ncol=len(current_states), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    plt.show()
    return fig, ax