from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import string
import plotly.express as px
import random
from scipy.stats import gaussian_kde
import utils

def play_solitaire():
    """
    Simulates a very simplified game:
    - We have a deck with cards from 1 to 13 (like suits).
    - We win if we draw at least one 'special' card (e.g., the number 7).
    """
    deck = list(range(1, 14))
    random.shuffle(deck)
    return 7 in deck[:5]  # returns a boolean

# Chain reaction (generation-based)
def simulate_chain_reaction_generational(max_generations=10):
    """
    Monte Carlo + Markov simulation:
    - Tracks neutrons generation by generation.
    - Each neutron may cause fission, scatter, or escape.
    - Fission produces 0–2 secondary neutrons (simplified).
    - Computes K_eff = average(neutrons_{t+1} / neutrons_t).
    """

    # --- MONTE CARLO PART ---
    # Randomly assign probabilities for this simulation
    p_fission = random.uniform(0.5, 0.7)
    p_scatter = random.uniform(0, 1 - p_fission)
    p_escape = 1 - p_fission - p_scatter

    # --- MARKOV CHAIN PART ---
    neutrons_per_generation = [10]  # start with 10 neutrons It stabilizes the simulation and reduces false extinction.
    K_values = []

    for gen in range(max_generations):
        active_neutrons = neutrons_per_generation[-1]
        if active_neutrons == 0:
            break

        new_neutrons = 0 ## re-starting neutrons for the next set of neutrons
        for _ in range(active_neutrons):
            r = random.random()
            if r < p_fission:
                # fission creates new neutrons (using avg of 2)
                new_neutrons += random.choices([1, 2],weights=[0.2,0.8])[0]
                
            elif r < p_fission + p_scatter:
                pass  # scattered, no new neutrons
            else:
                pass  # escaped, no new neutrons

        # Record this generation’s ratio (Markov transition)
        if active_neutrons > 0:
            K_values.append(new_neutrons / active_neutrons)

        neutrons_per_generation.append(new_neutrons)

    # Average K_eff over all valid generations
    K_eff = np.mean(K_values) if K_values else 0
    return K_eff >= 1, K_eff


##Visualization

def plot_k_eff_distribution(K_eff_estimates, bins=50):
    """
    Plots a histogram and KDE trend line for K_eff values, including mean and median markers.

    Parameters
    ----------
    K_eff_estimates : array-like
        Sequence of K_eff values to visualize.
    bins : int, optional
        Number of bins for the histogram (default: 50).
    """

    plt.figure(figsize=(8, 5))

    # Histogram
    count, bins, _ = plt.hist(
        K_eff_estimates,
        bins=bins,
        color='skyblue',
        edgecolor='black',
        alpha=0.7,
        density=True
    )

    # KDE trend line
    kde = gaussian_kde(K_eff_estimates)
    x_vals = np.linspace(min(K_eff_estimates), max(K_eff_estimates), 1000)
    plt.plot(x_vals, kde(x_vals), color='darkblue', linewidth=2, label='Trend (KDE)')

    # Mean and median lines
    mean_K = np.mean(K_eff_estimates)
    median_K = np.median(K_eff_estimates)
    plt.axvline(mean_K, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_K:.3f}")
    plt.axvline(median_K, color='green', linestyle='-.', linewidth=2, label=f"Median = {median_K:.3f}")

    # Labels and formatting
    plt.title("Distribution of Effective Multiplication Factor (Generation-based K_eff)")
    plt.xlabel("K_eff")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.4, linestyle='--')
    plt.show()

def plot_monte_carlo_results(results, all_simulations, N, block_size=1000):
    """
    Plots Monte Carlo convergence and block-wise probability distribution.

    Parameters
    ----------
    results : array-like
        Sequence of cumulative Monte Carlo estimates (e.g., running means).
    all_simulations : array-like
        Raw simulation outcomes (e.g., 0/1 results from each run).
    N : int
        Total number of simulations.
    block_size : int, optional
        Number of simulations per block for histogram calculation (default: 1000).
    """

    # Create a 1x2 subplot grid
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # --- [0] Convergence plot ---
    axs[0].plot(range(1, N + 1), results, label='Monte Carlo Estimate', color='blue')
    axs[0].axhline(
        y=results[-1],
        color='red',
        linestyle='--',
        label=f'Final value ≈ {results[-1]:.4f}'
    )
    axs[0].set_xlabel('Number of simulations')
    axs[0].set_ylabel('Estimated probability')
    axs[0].set_title('Monte Carlo Method Convergence')
    axs[0].legend()
    axs[0].grid(True)

    # --- [1] Histogram of block results ---
    # Split results into blocks
    blocks = np.array(all_simulations).reshape(-1, block_size)
    frequencies = blocks.sum(axis=1) / block_size  # average probability per block

    axs[1].hist(frequencies, bins=40, color='orange', edgecolor='black')

    # Mean and median
    mean = np.mean(frequencies)
    median = np.median(frequencies)
    axs[1].axvline(mean, color='red', linestyle='--', label=f'Mean ≈ {mean:.4f}')
    axs[1].axvline(median, color='orange', linestyle='--', label=f'Median ≈ {median:.4f}')

    axs[1].set_xlabel(f'Winning probability in {block_size}-simulation blocks')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Probability Distribution in Blocks')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

def montecarlo_area_below_function(a, n=10000, description=True, y_func=None):
    """
    Estimate the area under a given function using the Monte Carlo method.
    
    Parameters:
    a (float): The maximum absolute value for x (the function is evaluated from -a to a).
    n (int): The number of random points to generate for the calculation.
    description (bool): If True, prints and shows a description with the graph.
    y_func (callable): The function defining the curve. Default is y = x + b.
    b (float): A constant that modifies the default function (only used if y_func is None).
    
    Returns:
    float: The estimated area under the curve.
    """
    
    # Default function: y = x 
    if y_func is None:
        y_func = lambda x: x 

    ## GENERATE X AND Y VALUES FOR GRAPHING THE FUNCTION
    x = np.linspace(-a, a, n)  # x values between -a and a for plotting

    # Get the maximum y value of the function for the bounding box height
    max_y = np.max(y_func(x))

    # Area of the quadrilateral (bounding rectangle)
    area_quad = 2 * a * max_y  # width is 2a, height is max(y_func(x))

    ## CREATING RANDOM POINTS

    # Generate n random points for x between -a and a
    x_random = np.random.uniform(-a, a, n)

    # Generate n random points for y between 0 and max_y (the height of the bounding box)
    y_random = np.random.uniform(0, max_y, n)

    # Check if the random points are below the curve
    arr_below_curve = y_random <= y_func(x_random)

    # Count the number of points inside and outside the curve
    count_inside = arr_below_curve.sum()
    count_outside = arr_below_curve.size - count_inside

    # Estimate the area under the curve
    calculated_area = area_quad * (count_inside / (count_inside + count_outside))

    # SHOW THE GRAPH AND DESCRIPTION
    if description:
        # Plot the function curve
        plt.plot(x, y_func(x), label=f'Curve: y = f(x)', color='blue')

        # Plot the random points (red dots)
        plt.scatter(x_random, y_random, color='red', label='Random points', alpha=0.1)

        # Labels and title for the graph
        plt.xlabel('x')
        plt.ylabel('y = f(x)')
        plt.title('Graph of the function with random points')

        # Display grid and legend
        plt.grid(True)
        plt.legend()

        # Print out the results
        print(f"Out of {n} random points, {count_outside} are not under the curve.")
        print(f"{count_inside} points are under the curve.")
        print(f"Percentage of points under the curve: {(count_inside / (count_inside + count_outside)) * 100:.2f}%")
        print(f"Estimated area under the curve: {calculated_area:.4f}")

        # Show the plot
        plt.show()

    return calculated_area

def plot_simulation_histograms(prob_bag_C, prob_bag_V, bins=30):
    """
    Plots histograms for two sets of simulated probabilities (prob_bag_C and prob_bag_V)
    with their respective mean and median indicators.

    Parameters
    ----------
    prob_bag_C : array-like
        Simulated probabilities for bag C.
    prob_bag_V : array-like
        Simulated probabilities for bag V.
    bins : int, optional
        Number of bins for the histograms (default: 30).
    """

    # Compute statistics
    mean_C = np.mean(prob_bag_C)
    median_C = np.median(prob_bag_C)
    mean_V = np.mean(prob_bag_V)
    median_V = np.median(prob_bag_V)

    # Create 1x2 subplot grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- Histogram for prob_bag_C ---
    axes[0].hist(prob_bag_C, bins=bins, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(mean_C, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_C:.4f}')
    axes[0].axvline(median_C, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_C:.4f}')
    axes[0].set_title('Histogram of prob_bag_C')
    axes[0].set_xlabel('Probability of C')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # --- Histogram for prob_bag_V ---
    axes[1].hist(prob_bag_V, bins=bins, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(mean_V, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_V:.4f}')
    axes[1].axvline(median_V, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_V:.4f}')
    axes[1].set_title('Histogram of prob_bag_V')
    axes[1].set_xlabel('Probability of V')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

def plot_success_rate_convergence(success_rate, N, recent_window=1000, bins=30):
    """
    Visualizes the convergence and distribution of estimated success probabilities
    from a Monte Carlo or iterative simulation.

    Parameters
    ----------
    success_rate : array-like
        Sequence of cumulative success probabilities (e.g., running averages).
    N : int
        Total number of simulations.
    recent_window : int, optional
        Number of most recent estimates to include in the histogram (default: 1000).
    bins : int, optional
        Number of bins for the histogram (default: 30).
    """

    # Create a figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- (1) Line plot: convergence of probability ---
    axes[0].plot(range(1, N + 1), success_rate, color='blue')
    axes[0].axhline(
        y=success_rate[-1],
        color='red',
        linestyle='--',
        label=f'Final value ≈ {success_rate[-1]:.4f}'
    )
    axes[0].set_title("Convergence of Estimated Probability of Success")
    axes[0].set_xlabel("Number of Simulations")
    axes[0].set_ylabel("Estimated Probability")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # --- (2) Histogram: recent estimates ---
    recent_data = success_rate[-recent_window:] if len(success_rate) >= recent_window else success_rate
    axes[1].hist(recent_data, bins=bins, color='orange', alpha=0.7, edgecolor='black')
    axes[1].set_title(f"Distribution of Last {len(recent_data)} Probability Estimates")
    axes[1].set_xlabel("Probability")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Print final estimate
    final_prob = success_rate[-1]
    print(f"Final estimated probability of self-sustaining chain reaction ≈ {final_prob:.4f}")