import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

def simulate_batch_testing_trial(args):
    N, p, k = args
    infected = np.random.rand(N) < p
    
    num_tests = 0
    for i in range(0, N, k):
        batch = infected[i:i+k]
        num_tests += 1
        if np.any(batch): 
            num_tests += np.sum(batch)  # Test infected individuals in the batch
            
    return num_tests

def simulate_batch_testing_parallel(N, p, k, num_trials=500):
    with Pool(8) as pool:
        trials_per_worker = [(N, p, k)] * (num_trials // 8)
        results = pool.map(simulate_batch_testing_trial, trials_per_worker)
    
    return np.mean(results)

def find_optimal_batch_size(N, p, num_trials=500):
    lower, upper = 1, 100
    
    batch_sizes = [1]
    while batch_sizes[-1] * 2 <= upper:
        batch_sizes.append(batch_sizes[-1] * 2)
    
    expected_tests = []
    
    for k in batch_sizes:
        print(f"Simulating batch size k = {k}")
        expected_tests.append(simulate_batch_testing_parallel(N, p, k, num_trials))
    
    optimal_k = batch_sizes[np.argmin(expected_tests)]
    
    # Plotting the expected number of tests vs. batch size
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, expected_tests, label="Expected Tests", marker='o')
    plt.scatter(optimal_k, expected_tests[np.argmin(expected_tests)], color='red', label=f"Optimal k = {optimal_k}", zorder=5)
    plt.xlabel('Batch Size (k)')
    plt.ylabel('Expected Number of Tests')
    plt.title(f'Optimal Batch Size for p = {p}')
    plt.grid(True)
    plt.legend()
    plt.savefig('task1-montecarlo-batch-size.pdf')

    return optimal_k, expected_tests

def quantify_reduction_in_workload(N, p, optimal_k, num_trials=500):
    individual_tests = N
    expected_tests_optimal = simulate_batch_testing_parallel(N, p, optimal_k, num_trials)
    reduction = (individual_tests - expected_tests_optimal) / individual_tests * 100
    
    # Plotting the reduction in workload
    batch_sizes = [1]
    while batch_sizes[-1] * 2 <= 100:
        batch_sizes.append(batch_sizes[-1] * 2)
    
    reductions = []
    for k in batch_sizes:
        expected_tests_k = simulate_batch_testing_parallel(N, p, k, num_trials)
        reductions.append((individual_tests - expected_tests_k) / individual_tests * 100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, reductions, label="Workload Reduction (%)", marker='s', color='green')
    plt.axhline(y=0, color='black', linestyle='--', label="No Reduction")
    plt.xlabel('Batch Size (k)')
    plt.ylabel('Workload Reduction (%)')
    plt.title(f'Workload Reduction vs. Batch Size for p = {p}')
    plt.grid(True)
    plt.legend()
    plt.savefig('task1-montecarlo-reduction.pdf')
    
    return expected_tests_optimal, reduction

# Example usage:
N = 10**6  # 1 million samples
p = 0.01   # Probability of infection

optimal_k, expected_tests = find_optimal_batch_size(N, p)
expected_tests_optimal, reduction = quantify_reduction_in_workload(N, p, optimal_k)

print(f"Optimal batch size: {optimal_k}")
print(f"Expected number of tests with optimal batch size: {expected_tests_optimal}")
print(f"Reduction in workload: {reduction:.2f}%")
