import os
import numpy as np
from numpy.lib.format import open_memmap


def load_dataset(dataset_path, mode='r'):
    """
    Load OGBench dataset with appropriate memory-mapping based on data types.
    """
    dtype_map = {
        'observations': np.float32,
        'next_observations': np.float32,
        'actions': np.float32,
        'terminals': np.float32,
        'qpos': np.float32,
        'qvel': np.float32,
        'text_descriptions': 'U512',           # Unicode strings
        'description_embeddings': np.float32,
    }

    dataset = {}
    for key, dtype in dtype_map.items():
        file_path = os.path.join(dataset_path, f'{key}.npy')
        if not os.path.exists(file_path):
            continue
        try:
            dataset[key] = open_memmap(file_path, mode=mode, dtype=dtype)
        except Exception as e:
            print(f"Failed to load {key}.npy: {e}")
            raise e
    return dataset


def unzip_npz(npz_path, output_dir=None):
    """
    Extracts the contents of an .npz file into a directory with each key as a separate .npy file.

    Parameters:
    - npz_path: Path to the .npz file.
    - output_dir: Directory where the extracted files will be saved.
    """
    # Check if suffix is .npz if not add
    if output_dir is None:
        output_dir = npz_path

    if not npz_path.endswith('.npz'):
        npz_path += '.npz'
    # Check if the .npz file exists
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"The file {npz_path} does not exist.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the .npz file
    print(f"Extracting data from {npz_path}...")
    with np.load(npz_path) as data:
        # Iterate over each key in the .npz file
        for key in data.files:
            array = data[key]
            # Define the output file path
            output_file = os.path.join(output_dir, f"{key}.npy")
            # Save the array to the output file
            np.save(output_file, array)
            print(f"Saved {key} to {output_file}")

    print(f"All data extracted to {output_dir}")


def extract_next_observations(dataset_path):
    """Preprocess entire dataset once to create next_observations, etc."""
    print("Extracting next observations...")

    dataset = load_dataset(dataset_path, mode='r')

    ob_mask = (1.0 - dataset['terminals']).astype(bool)
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    dataset['next_observations'] = dataset['observations'][next_ob_mask]
    dataset['observations'] = dataset['observations'][ob_mask]
    dataset['actions'] = dataset['actions'][ob_mask]

    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

    dataset['qpos'] = dataset['qpos'][ob_mask]
    dataset['qvel'] = dataset['qvel'][ob_mask]

    # Save the updated dataset
    for key in dataset.keys():
        np.save(dataset_path + f'/{key}.npy', dataset[key])


if __name__ == "__main__":
    dataset_dir = os.path.expanduser('/.ogbench/data')
    dataset_name = 'visual-cube-single-play-v0'

    if dataset_name not in os.listdir(dataset_dir):
        import ogbench

        print(f"Downloading dataset: {dataset_name} to: {dataset_dir}")
        ogbench.download_datasets(dataset_name, dataset_dir=dataset_dir)

        dataset_path = os.path.join(dataset_dir, f"{dataset_name}")

        unzip_npz(npz_path=dataset_path)
        extract_next_observations(dataset_path=dataset_path)

        print("Dataset preprocessing complete.")
    else:
        print(f"Dataset {dataset_name} already exists in {dataset_dir}")
