import minari
import numpy as np
import pickle

# A list of d4rl datasets to convert to the pkl format
# The keys are the minari dataset names, and the values are the names for the output pkl files.
d4rl_datasets = {
    'mujoco/hopper/medium-v0': 'hopper-medium-v2.pkl',
    # 'hopper-medium-replay-v0' is not available in minari, so it's removed.
    'mujoco/hopper/expert-v0': 'hopper-expert-v2.pkl',
    'mujoco/halfcheetah/medium-v0': 'halfcheetah-medium-v2.pkl',
    # 'halfcheetah-medium-replay-v0' is not available in minari, so it's removed.
    'mujoco/halfcheetah/expert-v0': 'halfcheetah-expert-v2.pkl',
    'mujoco/walker2d/medium-v0': 'walker2d-medium-v2.pkl',
    # 'walker2d-medium-replay-v0' is not available in minari, so it's removed.
    'mujoco/walker2d/expert-v0': 'walker2d-expert-v2.pkl',
}

for minari_name, pkl_name in d4rl_datasets.items():	
    print(f'Processing {minari_name} -> {pkl_name}')

    # Load the dataset from minari
    # This will download the dataset if it's not already local
    dataset = minari.load_dataset(minari_name, download=True)

    paths = []
    for episode in dataset:
        # For each episode, construct the path dictionary in the format expected by the project
        
        # The 'terminals' array should be all False except for the last step
        terminals = np.zeros_like(episode.rewards, dtype=bool)
        terminals[-1] = True

        # The 'next_observations' array is the observation sequence shifted by one
        # For the last step, d4rl datasets often repeat the last observation, so we'll do the same.
        next_observations = np.concatenate([episode.observations[1:], episode.observations[-1:]], axis=0)

        path = {
            'observations': episode.observations,
            'actions': episode.actions,
            'rewards': episode.rewards,
            'next_observations': next_observations,
            'terminals': terminals,
        }
        paths.append(path)

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns):.2f}, std = {np.std(returns):.2f}, max = {np.max(returns):.2f}, min = {np.min(returns):.2f}')

    # Save the processed data to a pkl file
    with open(pkl_name, 'wb') as f:
        pickle.dump(paths, f)
    
    print(f'Successfully saved to {pkl_name}')
    print('=' * 50)

print("All datasets processed.")

