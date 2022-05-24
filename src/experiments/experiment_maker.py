import json
import itertools

options = {
    'model': ('UNet', 'UTAE', 'ViT'),
    'rgb_only': (True, False), 
    'multi_temporal': (True, False),
    'fold': (1, 2, 3, 4, 5),
}

# Get all possible combinations of the options
combinations = list(itertools.product(*options.values()))

# UNet can't be multi_temporal
combinations = [c for c in combinations if not ((c[0] in ('UNet',)) and c[2])]

# UTAE can't be non-multi_temporal
combinations = [c for c in combinations if not ((c[0] in ('UTAE',)) and not c[2])]

# All arguments
keys = list(options.keys())

# Create a dict with all the experiment arguments
all_options = {
    f'Experiment {i+1}': {keys[n]: c[n] for n in range(len(keys))} for i, c in enumerate(combinations)
}

# Write the experiments to a json file
with open('experiments/experiments.json', 'w') as f:
    json.dump(all_options, f, indent=4)