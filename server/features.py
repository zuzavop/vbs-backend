# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import torch

import numpy as np

from PIL import Image

import database as db
import configs as c
import logger as l
import models as m


def get_cosine_ranking(query_vector, matrix):
    # Get the dot product for every entry
    dot_product = torch.matmul(query_vector, matrix.T)

    # Sort for the indices of the nearest neighbors
    nearest_neighbors = torch.argsort(-dot_product)

    # Give back nearest neigbor sortings and distances
    return nearest_neighbors, dot_product


def get_images_by_text_query(query: str, k: int, dataset: str, model: str):
    start_time = time.time()

    # Tokenize query input and encode the data using the selected model
    text_features = m.embed_text(query, model)

    # Normalize vector to make it smaller and for cosine calculcation
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(text_features, data)
    sorted_indices = sorted_indices[:k]

    # If settings multiply and change to integer
    selected_data = data[sorted_indices]
    if c.BASE_MULTIPLICATION:
        selected_data = (selected_data * c.BASE_MULTIPLIER).int()

    l.logger.info(f'{db_time}')
    db_time = np.array(db_time)[sorted_indices].tolist()

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            selected_data.tolist(),
            labels[sorted_indices].tolist(),
            db_time,
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del similarities
    del sorted_indices
    del db_time

    # Return a list of (ID, rank, score, feature, label) tuples
    return most_similar_samples


def get_images_by_image_query(image: Image, k: int, dataset: str, model: str):
    start_time = time.time()

    # Preprocess query input and encode the data using the selected model
    image_features = m.embed_image(image, model)

    # Normalize vector to make it smaller and for cosine calculcation
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(image_features, data)
    sorted_indices = sorted_indices[:k]

    # If settings multiply and change to integer
    selected_data = data[sorted_indices]
    if c.BASE_MULTIPLICATION:
        selected_data = (selected_data * c.BASE_MULTIPLIER).int()

    db_time = np.array(db_time)[sorted_indices].tolist()

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            selected_data.tolist(),
            labels[sorted_indices].tolist(),
            db_time,
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del similarities
    del sorted_indices
    del db_time

    # Return a list of (ID, rank, score, feature, label) tuples
    return most_similar_samples


def get_images_by_image_id(id: str, k: int, dataset: str, model: str):
    start_time = time.time()

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()

    # Get the video id and frame_id
    video_id, frame_id = db.uri_spliter(id, dataset)
    id = video_id + db.name_splitter(dataset) + frame_id

    # Find the index of the provided 'id' within the 'ids' array
    idx = np.where(ids == id.encode('utf-8'))[0][0]
    image_features = data[idx]

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(image_features, data)
    sorted_indices = sorted_indices[:k]

    # If settings multiply and change to integer
    selected_data = data[sorted_indices]
    if c.BASE_MULTIPLICATION:
        selected_data = (selected_data * c.BASE_MULTIPLIER).int()

    db_time = np.array(db_time)[sorted_indices].tolist()

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            selected_data.tolist(),
            labels[sorted_indices].tolist(),
            db_time,
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del similarities
    del sorted_indices
    del db_time

    # Return a list of (ID, rank, score, feature, label) tuples
    return most_similar_samples


def get_video_images_by_id(id: str, k: int, dataset: str, model: str):
    # Load data from the database
    # Get an array of video IDs from the database
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()

    # Get the video id and frame_id
    video_id, frame_id = db.uri_spliter(id, dataset)
    id = video_id + db.name_splitter(dataset) + frame_id

    # Find the index of the provided 'id' within the 'ids' array
    condition = ids == id.encode('utf-8')
    if not np.any(condition):
        idx = 0
        while not ids[idx].decode('utf-8').startswith(video_id):
            idx += 1
    else:
        idx = np.where(condition)[0][0]

    # Set start and end indices
    if k == 0:
        k = 1000

    # Extract a slice of 'k' elements centered around the found index
    start_idx = idx - k
    end_idx = idx + k + 1

    # Get all video frames for the video
    if k == -1:
        # Get video frames before the provided
        cur_i = 0
        while ids[idx - cur_i].decode('utf-8').startswith(video_id):
            cur_i += 1
        start_idx = idx - cur_i + 1

        # Get video frames after the provided
        cur_i = 0
        while ids[idx + cur_i].decode('utf-8').startswith(video_id):
            cur_i += 1
        end_idx = idx + cur_i

    # Slice around start and end indices
    sliced_ids = ids[start_idx:end_idx]
    sliced_features = data[start_idx:end_idx]
    sliced_labels = labels[start_idx:end_idx]

    # If settings multiply and change to integer
    if c.BASE_MULTIPLICATION:
        sliced_features = (sliced_features * c.BASE_MULTIPLIER).int()

    db_time = np.array(db_time)[start_idx:end_idx].tolist()

    # Combine the selected IDs and features into a list of tuples
    video_images = list(
        zip(
            sliced_ids.tolist(),
            sliced_features.tolist(),
            sliced_labels.tolist(),
            db_time,
        )
    )

    del data
    del ids
    del labels
    del sliced_ids
    del sliced_features
    del sliced_labels
    del db_time

    # Return a list of (ID, feature, label) tuples
    return video_images


def get_random_video_frame(dataset: str, model: str):
    # Load data from the database
    # Get an array of video IDs from the database
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()

    # Generate a random index within the valid range of IDs
    random_id = np.random.randint(0, len(ids))

    # Select a single ID using the random index
    selected_ids = ids[random_id : random_id + 1]

    # Select the corresponding data or features using the random index
    selected_features = data[random_id : random_id + 1]

    # Select the corresponding data or features using the random index
    selected_labels = labels[random_id : random_id + 1]

    # If settings multiply and change to integer
    if c.BASE_MULTIPLICATION:
        selected_features = (selected_features * c.BASE_MULTIPLIER).int()

    db_time = np.array(db_time)[random_id : random_id + 1].tolist()

    # Combine the selected IDs and features into a list of tuples
    video_images = list(
        zip(
            selected_ids.tolist(),
            selected_features.tolist(),
            selected_labels.tolist(),
            db_time,
        )
    )

    del data
    del ids
    del labels
    del selected_ids
    del selected_features
    del selected_labels
    del db_time

    # Return a list of (ID, feature, label) tuples
    return video_images
