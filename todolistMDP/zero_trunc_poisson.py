import numpy as np

from collections import deque
from pprint import pprint
from scipy.stats import poisson


def bin_expected_value(prob, values):
    # Check whether prob and values have same length
    assert prob.shape == values.shape
    
    # Normalize probability
    prob /= np.sum(prob)
    
    # Compute & return expected value
    return np.sum(prob * values)


def compute_bin_means(bounds, pmf, values):
    # Initialize index bias
    shift = bounds[0]
    
    # Initialize bin lower bound
    low_idx = bounds[0] - shift
    
    # Initialize list of mean values
    means = deque()
    
    # For all bin bounds
    for bound in bounds[1:]:
        
        # Get bin upper interval bound
        high_idx = bound - shift + 1
        
        # Compute mean of the bin
        mean = bin_expected_value(prob=pmf[low_idx:high_idx],
                                  values=values[low_idx:high_idx])
        
        # Add mean to the list of mean values
        means.append(mean)
        
        # Move bin lower bound to the next bin
        low_idx = high_idx
    
    return np.array(means)


def compute_bin_probs(bounds, cdf):
    # Initialize index bias
    shift = bounds[0]
    
    # Initialize list of probabilities
    probs = deque()
    
    # Initialize cumulative probability
    cum_prob = 0
    
    for bound in bounds[1:]:
        # Get index of the bin upper bound
        idx = bound - shift
        
        # Compute bin probability
        bin_prob = cdf[idx] - cum_prob
        
        # Add probability to the list of probabilities
        probs.append(bin_prob)
        
        # Update cumulative probability
        cum_prob += bin_prob

    # Convert list to NumPy array
    probs = np.array(probs)

    # Check whether it is (close to) a valid probability distribution
    assert np.isclose(np.sum(probs), 1, atol=1e-9)
    
    # Normalize probability distribution
    probs /= np.sum(probs)
    
    return probs


def get_bin_intervals(lb, ub, num_bins=1):
    # Return bin intervals
    return np.unique(np.linspace(lb, ub, num_bins+1, dtype=np.int))


def get_binned_dist(mu, num_bins, std=3):
    
    if num_bins == 1:
        return {
            "bin_bounds":     np.array([1, np.PINF]),
            "bin_means":      np.array([mu]),
            "bin_probs":      np.array([1.]),
            "binned_mean":    mu,
        
            "distrib_cdf":    None,
            "distrib_mean":   mu,
            "distrib_pmf":    None,
            "distrib_values": None,
            "distrib_var":    mu,
        
            "range":          (1, np.PINF)
        }

    # Get lower and upper value of the interval that supports 3 stds
    low_idx, high_idx = get_std_indices(mu, std=std)
    
    # Get bin bounds
    bin_bounds = get_bin_intervals(low_idx, high_idx, num_bins=num_bins)
    
    # Distribution values
    distrib_values = np.arange(low_idx, high_idx + 1)
    
    # Compute probability mass function (pmf)
    distrib_pmf = pmf(distrib_values, mu)
    
    # Normalize distribution pmf
    distrib_pmf /= np.sum(distrib_pmf)
    
    # Compute cumulative distribution function
    distrib_cdf = np.cumsum(distrib_pmf)
    
    # Compute probability of each bin
    bin_probs = compute_bin_probs(bin_bounds, distrib_cdf)
    
    # Compute bin mean values
    bin_means = compute_bin_means(bin_bounds, distrib_pmf, distrib_values)
    
    binned_mean = bin_expected_value(bin_probs, bin_means)

    return {
        "bin_bounds":     bin_bounds,
        "bin_means":      bin_means,
        "bin_probs":      bin_probs,
        "binned_mean":    binned_mean,
        
        "distrib_cdf":    distrib_cdf,
        "distrib_mean":   mean(mu),
        "distrib_pmf":    distrib_pmf,
        "distrib_values": distrib_values,
        "distrib_var":    var(mu),
    
        "range": (low_idx, high_idx)
    }


def get_std_indices(mu, std=3):
    # Compute standard deviation of the distribution
    std_val = np.sqrt(var(mu))
    
    # Compute lower bound
    lower_bound = max(1, int(np.ceil(mu - std * std_val)))
    
    # Compute upper bound
    upper_bound = int(np.ceil(mu + std * std_val))
    
    return lower_bound, upper_bound


def mean(mu):
    return mu / (1 - np.exp(-mu))


def pmf(k, mu):
    k = np.array(k)
    if (k < 1).any():
        raise Exception(
            f"Invalid value {k} for zero-truncated Poisson distribution.")
    
    return poisson.pmf(k, mu) / (1 - poisson.pmf(0, mu))


def var(mu):
    return mean(mu) * (1 + mu - mean(mu))


if __name__ == '__main__':
    pprint(get_binned_dist(100, 2))
