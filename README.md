# discrete_markov: a Python library providing functions for working with discrete Markov Chains

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Markovkate_01.svg/1200px-Markovkate_01.svg.png" width="300" height="300">

Package discrete_markov provides functions for working with discrete Markov chains,
including calculating transition matrices, simulating Markov chains, and computing
stationary distributions.

Requirements: numpy, pandas, itertools

The functions offered are described below.

## Transition Matrix
The transition_matrix function computes the transition matrix from a given array of
states, and a given degree. The resulting transition matrix is a pandas DataFrame, 
where each row represents the current state and each column represents the next state.
The values in the matrix represent the probabilities of transitioning from the current
state to the next state, where rows represent previous/current state and columns are 
next states.

The degree indicates how many previous states to consider, default is 1.

## Simulate Markov Chain
The simulate_markov_chain function simulates the evolution of a Markov chain over a
specified time period, given the transition matrix and an initial state. It returns
an array of states representing the simulated trajectory of the Markov chain.

## Compute Stationary Distribution
The compute_stationary_distribution function computes the stationary distribution
matrix of a Markov chain, given a transition matrix of degree one. The stationary
distribution represents the long-term probabilities of being in each state of the
Markov chain. Note that convergence may not occur for some Markov chains, such as
absorbing, periodic, or non-ergodic chains.

## Visualize Transition Matrix
The visualize_transition_matrix function visualizes the Markov transition matrix using a horizontal bar chart. It takes a Pandas DataFrame representing the transition matrix as input, where each column and row label represents a state. The values in the matrix represent the probabilities of transitioning from the current state to the next state. The resulting plot provides an intuitive visualization of the transition probabilities between states.

## Limitations
The transition_matrix function is recommended for use for degrees up to approximately
five, as the number of rows and time complexity increase exponentially with higher 
degrees. The compute_stationary_distribution function requires a square transition 
matrix (n x n), and the rows and columns must have consistent naming. The visualize_transition_matrix
function is suitable for Markov transition matrices with a reasonable number of states. For large matrices, the plot may become crowded and less interpretable.


It is recommended to carefully review the documentation and ensure proper usage of
the functions in this package for accurate and reliable results in working with 
discrete Markov chains.
