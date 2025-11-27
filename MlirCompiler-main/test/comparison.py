#!/usr/bin/env python3
import subprocess
import re
import matplotlib.pyplot as plt
import statistics

def extract_time(output):
    """Extract time from program output"""
    patterns = [
        r'Computation time:\s*([\d.]+)\s*ms',
        r'OpenBLAS.*?:\s*([\d.]+)\s*ms',
        r'([\d.]+)\s*ms.*Result check',
        r'Time:\s*([\d.]+)\s*ms'
    ]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    return None

# Run benchmarks
print("Running OpenBLAS...")
openblas_times = []
for i in range(30):
    result = subprocess.run(
        ['taskset', '-c', '0-15', './openblas'], 
        capture_output=True, 
        text=True
    )
    time_val = extract_time(result.stdout)
    if time_val:
        openblas_times.append(time_val)
        print(f"  Run {i+1}: {time_val:.3f} ms")

print("\nRunning MLIR...")
mlir_times = []
for i in range(30):
    result = subprocess.run(
        ['taskset', '-c', '0-15', './matmul'], 
        capture_output=True, 
        text=True
    )
    time_val = extract_time(result.stdout)
    if time_val:
        mlir_times.append(time_val)
        print(f"  Run {i+1}: {time_val:.3f} ms")

# Calculate statistics
if openblas_times and mlir_times:
    openblas_mean = statistics.mean(openblas_times)
    mlir_mean = statistics.mean(mlir_times)
    ratio = mlir_mean / openblas_mean
    
    print(f"\nOpenBLAS: {openblas_mean:.2f} ms")
    print(f"MLIR: {mlir_mean:.2f} ms")
    print(f"Ratio: {ratio:.1f}x")
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Performance over time
    ax1.plot(range(1, len(openblas_times) + 1), openblas_times, 'o-', 
             label='OpenBLAS', color='blue', linewidth=2, markersize=6)
    ax1.plot(range(1, len(mlir_times) + 1), mlir_times, 'o-', 
             label='My MLIR', color='red', linewidth=2, markersize=6)
    ax1.set_xlabel('Run Number')
    ax1.set_ylabel('Computation Time (ms)')
    ax1.set_title('Performance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average performance bar chart
    labels = ['OpenBLAS', 'My MLIR']
    means = [openblas_mean, mlir_mean]
    colors = ['#1f77b4', '#d62728']
    
    bars = ax2.bar(labels, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Average Time (ms)')
    ax2.set_title(f'Average Performance\nMLIR is {ratio:.1f}x slower')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mean:.2f} ms', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('simple_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

else:
    print("Error: Could not extract timing data")