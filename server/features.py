# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import torch

import numpy as np
import pandas as pd

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


def get_images_by_text_query(query: str, k: int, dataset: str, model: str, selected_indices: list):
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
    
    if selected_indices is not None:
        data = data[selected_indices]
        ids = ids[selected_indices]
        labels = labels[selected_indices]
        if db_time.shape[0] > 0:
            db_time = db_time[selected_indices]

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


def get_images_by_image_query(image: Image, k: int, dataset: str, model: str, selected_indices: list):
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
    
    if selected_indices is not None:
        data = data[selected_indices]
        ids = ids[selected_indices]
        labels = labels[selected_indices]
        if db_time.shape[0] > 0:
            db_time = db_time[selected_indices]

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


def get_images_by_image_id(id: str, k: int, dataset: str, model: str, selected_indices: list):
    start_time = time.time()

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()
    
    if selected_indices is not None:
        data = data[selected_indices]
        ids = ids[selected_indices]
        labels = labels[selected_indices]
        if db_time.shape[0] > 0:
            db_time = db_time[selected_indices]

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
    start_idx = idx - int(k/2)
    end_idx = idx + int(k/2) + 1

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


def get_images_by_temporal_query(query: str, k: int, dataset: str, model: str, is_life_log: bool, selected_indices: list):
    start_time = time.time()
    
    queries = query.split(">")
    separator = db.name_splitter(dataset).encode()

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
    
    if selected_indices is not None:
        data = data[selected_indices]
        ids = ids[selected_indices]
        labels = labels[selected_indices]
        if db_time.shape[0] > 0:
            db_time = db_time[selected_indices]
        
    if is_life_log: 
        sequences = []
        separator_ids = [id.rpartition(separator)[0] for id in ids]
        for i, text_f in enumerate(text_features):
            _, sim = get_cosine_ranking(text_f, data)
            if i == 0:
                # For the first action, start new sequences
                sequences = [[(id, s)] for id, s in enumerate(sim)]
            else:
                # For subsequent actions, extend existing sequences
                new_sequences = []
                for seq in sequences:
                    seq_id = seq[-1][0]
                    seq_separator_id = separator_ids[seq_id]
                    for id, s in enumerate(sim):
                        if id > seq_id:
                            if separator_ids[id] == seq_separator_id:
                                new_seq = seq + [(id, s)]
                                new_sequences.append(new_seq)
                        elif separator_ids[id] != seq_separator_id:
                            break
                sequences = new_sequences
        
         # Sort the sequences by their total similarity score
        sequences.sort(key=lambda seq: -sum(s for _, s in seq))
    else:
        separator_ids = [id.rpartition(separator)[0] for id in ids]
        max_scores_per_day = {}
        
        for i, text_f in enumerate(text_features):
            _, sim = get_cosine_ranking(text_f, data)
            for id, s in enumerate(sim):
                day = separator_ids[id]
                if day not in max_scores_per_day:
                    max_scores_per_day[day] = [(id, s)]
                elif len(max_scores_per_day[day]) <= i:
                    max_scores_per_day[day].append((id, s))
                elif s > max_scores_per_day[day][i][1]:
                    max_scores_per_day[day][i] = (id, s)

        # Get the images with the maximum similarity score for each day
        max_images = [max_scores_per_day[day] for day in max_scores_per_day]

        # Sort the images by their similarity score
        max_images.sort(key=lambda img: -sum(s for _, s in img))
        sequences = [[img] for img in max_images]

    sorted_indices = sequences[:k]
    similarities = [s.item() for seq in sorted_indices for _, s in seq]
    sorted_indices = [idx for seq in sorted_indices for idx, _ in seq]

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
            similarities,
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


def weekday_to_number(weekday):
    weekdays = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}
    return weekdays.get(weekday, -1)


def get_filter_indices(filter: dict, dataset: str):
    # Load metadata from the database
    db.load_features(dataset, 'clip-vit-webli')
    metadata = db.get_metadata()
    
    # Initialize indices with all indices in metadata
    indices = metadata.index.tolist()
    
    # Iterate over the filter dictionary and apply each filter
    for column, value in filter.items():
        if column == "weekday" and isinstance(value, str):
            value = weekday_to_number(value)
        if column == "id" and value.startswith("yyyy"):
            value1 = value.replace("yyyy", "2019")
            filter_indices1 = metadata[metadata[column].str.startswith(value1)].index.tolist()
            value2 = value.replace("yyyy", "2020")
            filter_indices2 = metadata[metadata[column].str.startswith(value2)].index.tolist()
            filter_indices = filter_indices1 + filter_indices2
            indices = list(set(indices) & set(filter_indices))
            continue
        
        # Get the indices of the metadata that match the current filter
        if column == 'id':
            filter_indices = metadata[metadata[column].str.startswith(value)].index.tolist()
        elif column == "weekday":
            filter_indices = metadata[metadata[column] == value].index.tolist()
        else:
            filter_indices = metadata[metadata[column].str.contains(value, case=False)].index.tolist()
        
        # Intersect indices with filter_indices
        indices = list(set(indices) & set(filter_indices))
    
    return indices


def filter_metadata(filter: dict, k: int, dataset: str):
    # Load metadata from the database
    db.load_features(dataset, 'clip-vit-webli')
    ids = db.get_ids()
    data = db.get_data()
    labels = db.get_labels()
    db_time = db.get_time()
    
    selected_indices = get_filter_indices(filter, dataset)[:k]
    
    # If settings multiply and change to integer
    selected_data = data[selected_indices]

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, selected_indices, ids, dataset)

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[selected_indices].tolist(),
            [i for i in range(len(selected_indices))],
            np.zeros(len(selected_indices)).tolist(),
            selected_data.tolist(),
            labels[selected_indices].tolist(),
            db_time.tolist(),
        )
    )

    del data
    del ids
    del labels
    del db_time

    return most_similar_samples


def get_filters(dataset: str):
    # Load metadata from the database
    db.load_features(dataset, 'clip-vit-webli')
    metadata = db.get_metadata()

    # Get the list of available filters
    filters = metadata.columns.tolist()
    
    # Return the list of available filters
    return filters