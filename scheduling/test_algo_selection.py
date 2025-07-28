from load_CPU_data import get_aligned_datasets

# Test configuration with selected algorithms
config = {'algorithms': ['TempoSight', 'CEDL']}

# Try loading data with algorithm filtering
data = get_aligned_datasets('Alibaba', config=config)

# Print out what we got
print(f"\nAlgorithms loaded: {list(data.get('Alibaba', {}).keys())}")

# Check if we have machine data for each algorithm
for algo in data.get('Alibaba', {}):
    machine_count = len(data['Alibaba'][algo])
    print(f"  - {algo}: {machine_count} machines")

print("\nTest completed!")
