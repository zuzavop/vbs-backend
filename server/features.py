# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import torch

import numpy as np
import pandas as pd

from PIL import Image

from itertools import combinations_with_replacement

import database as db
import configs as c
import logger as l
import models as m


def normalize_vector(vector):
    # Normalize the vector by dividing it by its norm
    return vector / vector.norm(dim=-1, keepdim=True)


def load_data(dataset, model, selected_indices=None):
    # Load data from the database
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()
    
    # Filtered out data based on the selected indices
    if selected_indices is not None:
        data = data[selected_indices]
        ids = ids[selected_indices]
        labels = labels[selected_indices]
        if db_time.shape[0] > 0:
            db_time = db_time[selected_indices]
    
    return data, ids, labels, db_time


def fallback_time_stamps(ids, dataset):
    # Initialize an empty list to store time stamps
    time_stamps = []

    # Iterate through each ID in the given list
    for id in ids:
        # Convert the ID from bytes to string using UTF-8 encoding
        id = id.decode('utf-8')

        # Split the ID into two parts using a custom function (db.name_splitter) and get the second part
        _, frame_id = id.split(db.name_splitter(dataset), 1)
        if dataset == "MVK":
            frame_id = id.split("_")[-1]

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


def get_cosine_ranking(query_vector, matrix, top_k = -1):
    # Ensure the query vector has the same data type as the matrix
    query_vector = query_vector.to(matrix.dtype)
    
    num_samples = matrix.shape[0]
    if top_k == -1 or top_k > num_samples:
        top_k = num_samples
    
    # Get the dot product for every entry
    dot_product = torch.matmul(query_vector, matrix.T)
    
    # Get the indices of the top_k most similar vectors
    _, nearest_neighbors = torch.topk(dot_product, top_k, dim=-1)

    # Return the indices of the top_k most similar vectors and their similarity scores
    return nearest_neighbors, dot_product


def get_images_by_text_query(query: str, k: int, dataset: str, model: str, selected_indices: list):
    start_time = time.time()

    # Tokenize query input and encode the data using the selected model
    text_features = m.embed_text(query, model)

    # Normalize vector to make it smaller and for cosine calculcation
    text_features = normalize_vector(text_features)
    
    # Load data
    data, ids, labels, db_time = load_data(dataset, model, selected_indices)

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(text_features, data, top_k=k)

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
    image_features = normalize_vector(image_features)

    # Load data
    data, ids, labels, db_time = load_data(dataset, model, selected_indices)

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(image_features, data, top_k=k)

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

    # Get the video id and frame_id
    video_id, frame_id = db.uri_spliter(id, dataset)
    id = video_id + db.name_splitter(dataset) + frame_id

    # Find the index of the provided 'id' within the 'ids' array
    idx = np.where(ids == id.encode('utf-8'))[0][0]
    
    if selected_indices is not None:
        if idx not in selected_indices:
            selected_indices.append(idx)
        data = data[selected_indices]
        ids = ids[selected_indices]
        labels = labels[selected_indices]
        if db_time.shape[0] > 0:
            db_time = db_time[selected_indices]
    
    idx = np.where(ids == id.encode('utf-8'))[0][0]
    image_features = data[idx]

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(image_features, data, top_k=k)

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


def get_images_by_temporal_query(query: str, query2: str, k: int, dataset: str, model: str, is_life_log: bool, selected_indices: list):
    start_time = time.time()
    
    queries = [query, query2]

    # Tokenize query input and encode the data using the selected model
    text_features = [m.embed_text(q, model) for q in queries]

    # Normalize vector to make it smaller and for cosine calculcation
    text_features = [normalize_vector(text_feature) for text_feature in text_features]

    # Load data
    data, ids, labels, db_time = load_data(dataset, model, selected_indices)
        
    if is_life_log: 
        # Split the IDs into the days
        separator_ids = np.array([id[:11] for id in ids])
        split_points = np.r_[0, np.where(separator_ids[:-1] != separator_ids[1:])[0] + 1, len(separator_ids) - 1]
        
        # Get the maximum similarity and the index of the maximum similarity for each text feature
        max_values_and_indices = []
        for text_f in text_features:
            _, sim = get_cosine_ranking(text_f, data)
            sim = sim.numpy()
            max_values_and_indices.append([
                (sim[split_points[i]:split_points[i+1]].max(), sim[split_points[i]:split_points[i+1]].argmax() + split_points[i])
                for i in range(len(split_points) - 1)
            ])
        
        # Split the days into hours
        day_splits = [i + 1 for i in range(len(split_points) - 2) if separator_ids[split_points[i]][:8] != separator_ids[split_points[i + 1]][:8]]
        days = np.split(split_points, day_splits)
        day_splits = np.r_[0, day_splits]

        # Get the combinations of the indices for each day
        max_images = []
        for d, day in enumerate(days):
            day_start = day_splits[d]
            indices = np.arange(day_start, day_start + len(day))
            combinations = list(combinations_with_replacement(indices, len(queries)))
            
            sequences = []
            for combination in combinations:
                seq = [(max_values_and_indices[q][combination[q]][0], max_values_and_indices[q][combination[q]][1]) for q in range(len(queries)) if q < len(max_values_and_indices) and combination[q] < len(max_values_and_indices[q])]
                images = [image.item() for _, image in seq]
                if len(images) != len(set(images)):
                    continue
                sequences.append(seq)
            if len(sequences) == 0:
                continue
            max_images.append(max(sequences, key=lambda x: sum(s for s, _ in x)))
            
        # Sort the images by their similarity score
        max_images.sort(key=lambda img: -sum(s for s,_ in img))
    else:
        split_points = db.get_splits()
        
        # Get the maximum similarity and the index of the maximum similarity for each text feature
        max_values_and_indices = []
        for text_f in text_features:
            _, sim = get_cosine_ranking(text_f, data)
            sim = sim.numpy()
            max_values_and_indices.append([
                (sim[split_points[i]:split_points[i+1]].max(), sim[split_points[i]:split_points[i+1]].argmax() + split_points[i])
                for i in range(len(split_points) - 1)
            ])

        max_images = [[(max_values_and_indices[i][j][0], max_values_and_indices[i][j][1]) for i in range(len(max_values_and_indices))] for j in range(len(max_values_and_indices[0]))]
        # Sort the images by their similarity score
        max_images.sort(key=lambda img: -sum(s for s,_ in img))

    sorted_indices = max_images[:k]
    similarities = [s.item() for seq in sorted_indices for s, _ in seq]
    sorted_indices = [idx for seq in sorted_indices for _, idx in seq]
    
    # Add a new value between each pair of consecutive indices
    extended_indices = []
    extended_similarities = []
    for i in range(0, len(sorted_indices), 2):
        extended_indices.append(sorted_indices[i])
        extended_similarities.append(similarities[i])
        middle_value = (sorted_indices[i] + sorted_indices[i + 1]) // 2
        extended_indices.append(middle_value)
        extended_similarities.append(0)
        extended_indices.append(sorted_indices[i + 1])
        extended_similarities.append(similarities[i + 1])
        last_value = sorted_indices[i + 1] - 1 if (sorted_indices[i] > sorted_indices[i + 1]) else sorted_indices[i + 1] + 1
        extended_indices.append(last_value)
        extended_similarities.append(0)

    
    # If settings multiply and change to integer
    selected_data = data[extended_indices]
    if c.BASE_MULTIPLICATION:
        selected_data = (selected_data * c.BASE_MULTIPLIER).int()

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, extended_indices, ids, dataset)

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[extended_indices].tolist(),
            [i for i in range(len(extended_indices))],
            extended_similarities,
            selected_data.tolist(),
            labels[extended_indices].tolist(),
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
    weekdays = {
        "Monday": 0, "Mon": 0, "Mo": 0,
        "Tuesday": 1, "Tue": 1, "Tu": 1,
        "Wednesday": 2, "Wed": 2, "We": 2,
        "Thursday": 3, "Thu": 3, "Th": 3,
        "Friday": 4, "Fri": 4, "Fr": 4,
        "Saturday": 5, "Sat": 5, "Sa": 5,
        "Sunday": 6, "Sun": 6, "Su": 6,
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6
    }
    return weekdays.get(weekday, -1)


def apply_weekday_filter(metadata, column, value):
    value = weekday_to_number(value)
    return metadata[metadata[column] == value].index.tolist()


def apply_id_filter(metadata, column, value):
    if value.startswith("yyyymm"):
        value = value[6:]
        return metadata[metadata[column].str.slice(6, 8) == value].index.tolist()
    elif value.startswith("yyyy"):
        value1 = value.replace("yyyy", "2019")
        filter_indices1 = metadata[metadata[column].str.startswith(value1)].index.tolist()
        value2 = value.replace("yyyy", "2020")
        filter_indices2 = metadata[metadata[column].str.startswith(value2)].index.tolist()
        return filter_indices1 + filter_indices2
    else:
        return metadata[metadata[column].str.startswith(value)].index.tolist()
    

def apply_hour_filter(metadata, column, value):
    value1, value2 = value.split("-")
    filter_indices = []
    for i in range(int(value1), int(value2)):
        filter_indices += metadata[metadata[column].str.startswith(str(i).zfill(2))].index.tolist()
    return filter_indices


def apply_generic_filter(metadata, column, value):
    return metadata[metadata[column].str.contains(value, case=False)].index.tolist()


def get_filter_indices(filter: dict, dataset: str):
    # Load metadata from the database
    db.load_features(dataset)
    metadata = db.get_metadata()
    
    # Initialize indices with all indices in metadata
    indices = metadata.index.tolist()
    
    # Iterate over the filter dictionary and apply each filter
    for column, value in filter.items():
        if column == "weekday" and isinstance(value, str):
            filter_indices = apply_weekday_filter(metadata, column, value)
        elif column == "id":
            filter_indices = apply_id_filter(metadata, column, value)
        elif column == "hour" and "-" in value:
            filter_indices = apply_hour_filter(metadata, column, value)
        elif column == "weekday":
            filter_indices = metadata[metadata[column] == value].index.tolist()
        else:
            filter_indices = metadata[metadata[column].str.contains(value, case=False)].index.tolist()
        
        # Intersect indices with filter_indices
        indices = list(set(indices) & set(filter_indices))
    
    return indices


def filter_metadata(filter: dict, k: int, dataset: str):
    # Load metadata from the database
    db.load_features(dataset)
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
    db.load_features(dataset)
    metadata = db.get_metadata()
    
    # Return the list of available filters
    return metadata.columns.tolist()