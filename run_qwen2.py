import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Qwen2-VL imports
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from numpy.lib.format import open_memmap

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import argparse
from pathlib import Path

# Default values
DEFAULT_DATASET_NAME = 'visual-cube-single-play-v0'
DEFAULT_DATASET_DIR = Path('~/.ogbench/data')
DEFAULT_DATASET_SPLIT = 'train'
DEFAULT_VL_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct-AWQ"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_TRANSITIONS = 10


def visualize_clusters(dataset_path, batch_size=5000):
    dataset = load_dataset(dataset_path, mode='r')
    indices = np.random.choice(len(dataset[list(dataset.keys())[0]]), batch_size, replace=False)

    text_descriptions = dataset['descriptions'][indices]
    text_embeddings = dataset['description_embeddings'][indices]

    # Normalize embeddings for cosine similarity
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    # Step 1: Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(text_embeddings)

    # Step 2: Agglomerative Clustering with Distance Threshold
    distance_threshold = 20  # Similarity threshold; lower = finer clusters
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    # Note: Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    labels = clustering_model.fit_predict(distance_matrix)

    # Step 3: Map clusters to text descriptions and compute cluster insights
    cluster_texts = {i: [] for i in np.unique(labels)}
    cluster_similarities = {i: [] for i in np.unique(labels)}

    for i, cluster_id in enumerate(labels):
        cluster_texts[cluster_id].append(text_descriptions[i])
        cluster_similarities[cluster_id].append(similarity_matrix[i, labels == cluster_id])

    # Print cluster insights
    for cluster_id, sentences in cluster_texts.items():
        print(f"\nCluster {cluster_id + 1} ({len(sentences)} sentences):")
        print("Sample sentences:")
        for sentence in sentences[:5]:  # Show only the first 5 sentences
            print(f" - {sentence} \n")
        avg_similarity = np.mean(cluster_similarities[cluster_id])
        print(f"Average intra-cluster similarity: {avg_similarity:.4f}")

    # Step 4: Visualize clusters using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    embeddings_2d = tsne.fit_transform(text_embeddings)

    plt.figure(figsize=(8, 6))
    for cluster_id in np.unique(labels):
        plt.scatter(
            embeddings_2d[labels == cluster_id, 0],
            embeddings_2d[labels == cluster_id, 1], label=f'Cluster {cluster_id + 1}')
    plt.legend()
    plt.title(f'Agglomerative Clustering with Semantic Similarity (Threshold={distance_threshold})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


def load_dataset(dataset_path, mode='r'):
    """
    Load OGBench dataset with appropriate memory-mapping based on data types.
    """
    dtype_map = {'observations': np.float32,
                 'next_observations': np.float32,
                 'actions': np.float32,
                 'terminals': np.float32,
                 'qpos': np.float32,
                 'qvel': np.float32,
                 'descriptions': 'U512',
                 }

    dataset = {}
    for key, dtype in dtype_map.items():
        try:
            file_path = os.path.join(dataset_path, f'{key}.npy')
            file_path = os.path.expanduser(file_path)
            dataset[key] = open_memmap(file_path, mode=mode, dtype=dtype)
        except Exception as e:
            print(f"Failed to load {key}.npy: {e}")
    return dataset


def get_batch(dataset, start_idx, end_idx):
    batch = {}
    for key in dataset.keys():
        # For keys that are 2D (like observations) or 1D (like terminals), slice them.
        batch[key] = np.array(dataset[key][start_idx:end_idx])

    # Initialize placeholders for text info
    batch['descriptions'] = np.empty(end_idx - start_idx, dtype='U512')
    batch['descriptions'][:] = ""
    return batch


def create_new_dataset(N, mock_batch, dataset_path):
    new_dataset_path = dataset_path + '-enriched'
    new_dataset_path = os.path.expanduser(new_dataset_path)
    if not os.path.exists(new_dataset_path):
        os.makedirs(new_dataset_path)
    dataset = {}
    for key in mock_batch.keys():
        dataset[key] = open_memmap(f'{new_dataset_path}/{key}.npy', mode='w+', shape=(N,) + mock_batch[key].shape[
                                                                                            1:], dtype=mock_batch[
            key].dtype)
    return dataset


def numpy_to_pil_image(obs_frame):
    """Convert a [H,W,C] NumPy array to a PIL Image."""
    if obs_frame.dtype == np.uint8:
        pil_img = Image.fromarray(obs_frame)
    else:
        pil_img = Image.fromarray(obs_frame.astype(np.uint8))
    return pil_img


def create_diff_image(frame1, frame2):
    """
    Create a difference image highlighting changes between two frames (PIL or numpy).
    """
    import numpy as np
    from PIL import Image, ImageChops

    if isinstance(frame1, Image.Image) and isinstance(frame2, Image.Image):
        diff_pil = ImageChops.difference(frame1.convert('RGB'), frame2.convert('RGB'))
        return diff_pil
    else:
        diff_arr = np.abs(frame1.astype('int32') - frame2.astype('int32')).clip(0, 255).astype(np.uint8)
        diff_pil = Image.fromarray(diff_arr, 'RGB')
        return diff_pil


def generate_new_dataset(dataset_name='visual-cube-single-play-v0',
                         dataset_dir='/.ogbench/data',
                         dataset='train',
                         vl_model_name="Qwen/Qwen2-VL-7B-Instruct-AWQ",
                         batch_size=1,
                         max_transitions=1000,
                         ):
    """
    Preprocess the visual-cube-single-play-v0 dataset with textual descriptions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = dataset_name.split('-')
    env_name = '-'.join(splits[:-2] + splits[-1:])
    if dataset == 'train':
        dataset_path = os.path.join(dataset_dir, f'{dataset_name}')
    elif dataset == 'val':
        dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val')

    # 1. Load dataset with memory mapping
    dataset = load_dataset(dataset_path, mode='r')

    if max_transitions == -1:
        max_transitions = dataset['observations'].shape[0]
    N = min(max_transitions, dataset['observations'].shape[0])

    # 2. Load Qwen2-VL model
    model_name = vl_model_name
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    mock_batch = get_batch(dataset, 0, batch_size)
    new_dataset = create_new_dataset(N, mock_batch, dataset_path)

    # 4. Generate textual descriptions
    with tqdm(total=N, desc="Generating descriptions", unit="transitions") as pbar:
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            batch = get_batch(dataset, start_idx, end_idx)

            messages_list = []
            for i in range(end_idx - start_idx):
                pil_img_1 = numpy_to_pil_image(batch['observations'][i])
                pil_img_2 = numpy_to_pil_image(batch['next_observations'][i])
                actions_str = ', '.join(f'{a:.2f}' for a in batch['actions'][i])
                pil_diff = create_diff_image(pil_img_1, pil_img_2)

                user_text = ("Attached are three images:\n"
                             "1) First frame (initial state)\n"
                             "2) Second frame (final state)\n"
                             "3) Difference image highlighting pixel changes\n\n"
                             f"Action vector that transformed the scene: {actions_str}\n\n"

                             "Provide a single-sentence **imperative instruction** to move the scene from the initial to the final state.\n"

                             "Constraints:\n"
                             "- **Precision**: Use numeric values (e.g., +0.15 in x). No vague terms like 'slightly.'\n"
                             "- **Single sentence**\n"
                             "- **Coordinate references**: Use x, y, z (and additional coords if needed) to specify directions.\n"
                             "- **Objects**: Explicitly name which object moves (manipulator arm or red cube).\n"
                             "- **Imperative style**: Example: 'Move manipulator arm left by -0.15 in x.'\n\n"

                             "Preferred response format: '[Verb] [object] [direction(s)] by [+/-X.XX] in [axes].'\n"
                             "Examples:\n"
                             "- 'Move arm left by -0.15 in x and forward by +0.50 in y'\n"
                             "- 'Raise red cube upward by +0.10 in z'\n"
                             "- 'Reposition arm rightward by +1.20 in x and forward by +0.75 in y'\n")

                messages = [{"role": "user", "content": [{"type": "text", "text": user_text},
                                                         {"type": "image", "image": pil_img_1},
                                                         {"type": "image", "image": pil_img_2},
                                                         {"type": "image", "image": pil_diff}, ]}]

                messages_list.append(messages)

            # Qwen2-VL preprocessing
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in
                     messages_list]
            image_inputs = []
            for m in messages_list:
                img_in, _ = process_vision_info(m)
                image_inputs.append(img_in)

            inputs = processor(text=texts, images=image_inputs, videos=None, padding=True, return_tensors="pt")
            inputs = inputs.to(device)

            # Generate text
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            batch_descriptions = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            batch['descriptions'] = batch_descriptions

            # Save to new dataset
            for key in new_dataset.keys():
                new_dataset[key][start_idx:end_idx][:] = batch[key][:]
                new_dataset[key].flush()

            pbar.update(end_idx - start_idx)


def get_parser() -> argparse.ArgumentParser:
    """
    Creates and returns an argument parser for the dataset preprocessing script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Preprocess dataset for cube environment.")
    parser.add_argument(
        '--dataset_name',
        type=str,
        default=DEFAULT_DATASET_NAME,
        help='Name of the dataset to preprocess.'
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help='Directory to save the dataset.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['train', 'val'],
        default=DEFAULT_DATASET_SPLIT,
        help='Dataset split to preprocess ("train" or "val").'
    )
    parser.add_argument(
        '--vl_model_name',
        type=str,
        default=DEFAULT_VL_MODEL_NAME,
        help='Name of the Qwen2-VL model to use.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='Batch size for generating descriptions.'
    )
    parser.add_argument(
        '--max_transitions',
        type=int,
        default=DEFAULT_MAX_TRANSITIONS,
        help='Maximum number of transitions to preprocess.'
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfg_dict = vars(args)

    dataset_dir = cfg_dict['dataset_dir']
    splits = cfg_dict['dataset_name'].split('-')
    env_name = '-'.join(splits[:-2] + splits[-1:])
    if cfg_dict['dataset'] == 'train':
        dataset_path = os.path.join(dataset_dir, f"{cfg_dict['dataset_name']}")
    elif cfg_dict['dataset'] == 'val':
        dataset_path = os.path.join(dataset_dir, f"{cfg_dict['dataset_name']}-val")

    generate_new_dataset(**cfg_dict)

    # Visualize clusters
    # visualize_clusters(dataset_path + '-enriched', batch_size=5_000)


    # load enriched dataset
    # dataset = load_dataset(dataset_path + '-enriched', mode='r')
    # print(dataset.keys())