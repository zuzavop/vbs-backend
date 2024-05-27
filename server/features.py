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


def fallback_time_stamps(ids, dataset):
    # Initialize an empty list to store time stamps
    time_stamps = []

    # Iterate through each ID in the given list
    for id in ids:
        # Convert the ID from bytes to string using UTF-8 encoding
        id = id.decode('utf-8')

        # Split the ID into two parts using a custom function (db.name_splitter) and get the second part
        _, frame_id = id.split(db.name_splitter(dataset), 1)

        # Calculate the middle time based on the frame ID, assuming a frame rate of 30 frames per second
        middle_time = float(frame_id) / 30 * 1000

        # Create a list containing the ID, middle time, and a time range around the middle time
        time_stamps.append(
            [id, middle_time, max(0, middle_time - 1000), middle_time + 1000]
        )

    # Convert the list of time stamps to a NumPy array and return it
    return np.array(time_stamps)


def get_time_stamps(db_time, slicing, ids, dataset):
    # Check if the given db_time is empty
    if len(db_time) < 1:
        # If empty, use the fallback_time_stamps function to generate time stamps for the specified IDs and dataset
        db_time = fallback_time_stamps(ids[slicing], dataset)
    else:
        # If not empty, slice the db_time array based on the provided slicing parameter
        db_time = db_time[slicing]

    # Return the resulting db_time array (either from the fallback or sliced original)
    return db_time


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

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, sorted_indices, ids, dataset)

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            selected_data.tolist(),
            labels[sorted_indices].tolist(),
            db_time.tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del db_time
    del similarities
    del sorted_indices

    # Return a list of (ID, rank, score, feature, label, time) tuples
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

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, sorted_indices, ids, dataset)

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            selected_data.tolist(),
            labels[sorted_indices].tolist(),
            db_time.tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del db_time
    del similarities
    del sorted_indices

    # Return a list of (ID, rank, score, feature, label, time) tuples
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

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, sorted_indices, ids, dataset)

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            selected_data.tolist(),
            labels[sorted_indices].tolist(),
            db_time.tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del db_time
    del similarities
    del sorted_indices

    # Return a list of (ID, rank, score, feature, label, time) tuples
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

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, list(range(start_idx, end_idx)), ids, dataset)

    # Combine the selected IDs and features into a list of tuples
    video_images = list(
        zip(
            sliced_ids.tolist(),
            sliced_features.tolist(),
            sliced_labels.tolist(),
            db_time.tolist(),
        )
    )

    del data
    del ids
    del labels
    del db_time
    del sliced_ids
    del sliced_features
    del sliced_labels

    # Return a list of (ID, feature, label, time) tuples
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

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(
        db_time, list(range(random_id, random_id + 1)), ids, dataset
    )

    # Combine the selected IDs and features into a list of tuples
    video_images = list(
        zip(
            selected_ids.tolist(),
            selected_features.tolist(),
            selected_labels.tolist(),
            db_time.tolist(),
        )
    )

    del data
    del ids
    del labels
    del db_time
    del selected_ids
    del selected_features
    del selected_labels

    # Return a list of (ID, feature, label, time) tuples
    return video_images


def get_video_by_id(video_id: str, dataset: str):
    # TODO
    print(video_id, dataset)
    return None


def get_images_by_temporal_query(query: str, k: int, dataset: str, model: str, is_life_log: bool):
    start_time = time.time()
    
    queries = query.split("<")
    separator = db.name_splitter(dataset)

    # Tokenize query input and encode the data using the selected model
    text_features = [m.embed_text(q, model) for q in queries]

    # Normalize vector to make it smaller and for cosine calculcation
    text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()
        
    if is_life_log: 
        print() #TODO
    else: # Temporal query where all queries should be in the same order TODO
        sequences = []
        for i, text_f in enumerate(text_features):
            _, sim = get_cosine_ranking(text_f, data)
            if i == 0:
                # For the first action, start new sequences
                sequences = [[(id, s)] for id, s in enumerate(sim)]
            else:
                # For subsequent actions, extend existing sequences
                new_sequences = []
                for seq in sequences:
                    for id, s in enumerate(sim):
                        if id > seq[-1][0]  and ids[id].rpartition(separator)[0] == ids[seq[-1][0]].rpartition(separator)[0]:
                            new_seq = seq + [(id, s)]
                            new_sequences.append(new_seq)
                        elif ids[id].rpartition(separator)[0] != ids[seq[-1][0]].rpartition(separator)[0]:
                            break
                sequences = new_sequences
        
         # Sort the sequences by their total similarity score
        sequences.sort(key=lambda seq: -sum(s for _, s in seq))
        
    sorted_indices = sorted_indices[:k]

    # If settings multiply and change to integer
    selected_data = data[sorted_indices]
    if c.BASE_MULTIPLICATION:
        selected_data = (selected_data * c.BASE_MULTIPLIER).int()

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, sorted_indices, ids, dataset)

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities[sorted_indices].tolist(),
            selected_data.tolist(),
            labels[sorted_indices].tolist(),
            db_time.tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del labels
    del db_time
    del similarities
    del sorted_indices

    # Return a list of (ID, rank, score, feature, label, time) tuples
    return most_similar_samples


def filter_metadata(query: str, metadata_type: str, dataset: str):
    # Load metadata from the database
    db.load_metadata(dataset)
    metadata = db.get_metadata()
    ids = db.get_ids()

    # Get the IDs of the matching metadata
    matching_ids = ids[metadata[metadata_type] == query]

    # Return the list of matching IDs
    return matching_ids.tolist()


def get_filters(dataset: str):
    # Load metadata from the database
    db.load_metadata(dataset)
    metadata = db.get_metadata()

    # Get the unique values for each metadata type
    unique_values = {
        'action': metadata['action'].unique().tolist(),
        'object': metadata['object'].unique().tolist(),
        'location': metadata['location'].unique().tolist(),
        'time': metadata['time'].unique().tolist(),
    }

    # Return the dictionary of unique metadata values
    return unique_values