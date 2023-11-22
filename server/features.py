import h5py
import time
import torch
import numpy as np

from PIL import Image

from scipy.spatial.distance import cosine

import database as db
import logger as l
import model as m


def get_cosine_ranking(query_vector, matrix):
    # Get the dot product for every entry
    dot_product = torch.matmul(query_vector, matrix.T)

    # Sort for the indices of the nearest neighbors
    nearest_neighbors = torch.argsort(-dot_product)

    return nearest_neighbors, dot_product


def get_images_by_text_query(
    query: str, k: int, dataset: str, model: str, rounding: bool = False
):
    start_time = time.time()

    # Tokenize query input and encode the data using the selected model
    text_features = m.embed_text(query, model)

    # Normalize vector to make it smaller and for cosine calculcation
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    if rounding:
        data = data.round(8)
    ids = np.array(db.get_ids())
    labels = db.get_labels()

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(text_features, data)
    sorted_indices = sorted_indices[:k]

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            data[sorted_indices].tolist(),
            labels[sorted_indices].tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del similarities
    del sorted_indices

    return most_similar_samples


def get_images_by_image_query(
    image: Image, k: int, dataset: str, model: str, rounding: bool = False
):
    start_time = time.time()

    # Preprocess query input and encode the data using the selected model
    image_features = m.embed_image(image, model)

    # Normalize vector to make it smaller and for cosine calculcation
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    if rounding:
        data = data.round(8)
    ids = np.array(db.get_ids())
    labels = db.get_labels()

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(image_features, data)
    sorted_indices = sorted_indices[:k]

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            data[sorted_indices].tolist(),
            labels[sorted_indices].tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del similarities
    del sorted_indices

    return most_similar_samples


def get_images_by_image_id(
    id: str, k: int, dataset: str, model: str, rounding: bool = False
):
    start_time = time.time()

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    if rounding:
        data = data.round(8)
    ids = np.array(db.get_ids())
    labels = db.get_labels()

    # Find the index of the provided 'id' within the 'ids' array
    idx = np.where(ids == id.encode('utf-8'))[0][0]
    image_features = data[idx]

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(image_features, data)
    sorted_indices = sorted_indices[:k]

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            data[sorted_indices].tolist(),
            labels[sorted_indices].tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del similarities
    del sorted_indices

    return most_similar_samples


def get_video_images_by_id(
    id: str, k: int, dataset: str, model: str, rounding: bool = False
):
    # Load data from the database
    # Get an array of video IDs from the database
    db.load_features(dataset, model)
    data = db.get_data()
    if rounding:
        data = data.round(8)
    ids = np.array(db.get_ids())
    labels = db.get_labels()

    # Find the index of the provided 'id' within the 'ids' array
    idx = np.where(ids == id.encode('utf-8'))[0][0]

    # Extract a slice of 'k' elements centered around the found index
    ids = ids[idx - k : idx + k]
    sliced_features = data[idx - k : idx + k]

    # Combine the selected IDs and features into a list of tuples
    video_images = list(zip(ids.tolist(), sliced_features.tolist()))

    # Return a list of (ID, feature) pairs
    return video_images


def get_random_video_frame(dataset: str, model: str, rounding: bool = False):
    # Load data from the database
    # Get an array of video IDs from the database
    db.load_features(dataset, model)
    data = db.get_data()
    if rounding:
        data = data.round(8)
    ids = np.array(db.get_ids())

    # Generate a random index within the valid range of IDs
    random_id = np.random.randint(0, len(ids))

    # Select a single ID using the random index
    selected_ids = ids[random_id : random_id + 1]

    # Select the corresponding data or features using the random index
    selected_features = data[random_id : random_id + 1]

    # Combine the selected IDs and features into a list of tuples
    video_images = list(zip(selected_ids.tolist(), selected_features.tolist()))

    # Return a list of (ID, feature) pairs
    return video_images
