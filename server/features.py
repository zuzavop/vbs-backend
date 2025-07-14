# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import torch
from torch_scatter import scatter_max

import numpy as np
import pandas as pd

from PIL import Image

from itertools import combinations_with_replacement

import database as db
import configs as c
import logger as l
import models as m


def fallback_time_stamps(ids, dataset):
    name_split = db.name_splitter(dataset)

    # Initialize an empty list to store time stamps
    time_stamps = []

    # Iterate through each ID in the given list
    for id in ids:
        # Convert the ID from bytes to string using UTF-8 encoding
        id = id.decode('utf-8')

        # Split the ID into two parts using a custom function (db.name_splitter) and get the second part
        _, frame_id = id.split(name_split, 1)
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
        return fallback_time_stamps(ids[slicing], dataset)
    else:
        # If not empty, slice the db_time array based on the provided slicing parameter
        return db_time[slicing]


@torch.compile
def fast_matmul(q, dT):
    return q @ dT


def get_cosine_ranking(query_vector, matrix, top_k = -1, batch_size=9000000):
    query_vector = query_vector.to(matrix.dtype)

    num_samples = matrix.shape[0]
    if top_k == -1 or top_k > num_samples:
        top_k = num_samples

    dot_product = torch.matmul(query_vector, matrix.T)

    # Sort the concatenated results to get the top k most similar samples
    _, nearest_neighbors = torch.topk(dot_product, top_k, dim=-1)

    # Give back nearest neigbor sortings and distances
    return nearest_neighbors, dot_product


def get_texture_distance(query_vector, matrix, top_k = -1, batch_size=9000000):
    query_vector = query_vector.to(matrix.dtype)

    num_samples = matrix.shape[0]
    if top_k == -1 or top_k > num_samples:
        top_k = num_samples


    query_part1 = query_vector[:16]
    matrix_part1 = matrix[:, :16]
    query_part1_norm = query_part1 / torch.norm(query_part1, p=2)
    matrix_part1_norm = matrix_part1 / torch.norm(matrix_part1, p=2, dim=1, keepdim=True)

    # Normalize the second part (remaining features)
    query_part2 = query_vector[16:]
    matrix_part2 = matrix[:, 16:]
    query_part2_norm = query_part2 / torch.norm(query_part2, p=2)
    matrix_part2_norm = matrix_part2 / torch.norm(matrix_part2, p=2, dim=1, keepdim=True)

    # Compute Euclidean distances
    distance_part1 = torch.norm(matrix_part1_norm - query_part1_norm, p=2, dim=1)
    distance_part2 = torch.norm(matrix_part2_norm - query_part2_norm, p=2, dim=1)

    # Combine the distances
    total_distance = distance_part1 + distance_part2


    _, nearest_neighbors = torch.topk(-total_distance, top_k, dim=-1)

    # Give back nearest neigbor sortings and distances
    return nearest_neighbors, total_distance #, all_similarities


def get_images_by_text_query(query: str, k: int, dataset: str, model: str, selected_indices: list):
    # Tokenize query input and encode the data using the selected model
    text_features = m.embed_text(query, model)

    # Normalize vector to make it smaller and for cosine calculcation
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

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
            range(len(sorted_indices)),
            similarities[sorted_indices].tolist(),
            selected_data.tolist(),
            labels[sorted_indices].tolist(),
            db_time.tolist(),
        )
    )

    # Return a list of (ID, rank, score, feature, label, time) tuples
    return most_similar_samples


def get_images_by_image_query(image: Image, k: int, dataset: str, model: str, selected_indices: list):
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

    # Return a list of (ID, rank, score, feature, label, time) tuples
    return most_similar_samples


def get_images_by_image_id(id: str, k: int, dataset: str, model: str, selected_indices: list):
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
    sorted_indices, similarities = get_cosine_ranking(image_features, data, top_k = k)

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

    # Return a list of (ID, rank, score, feature, label, time) tuples
    return most_similar_samples


def get_video_images_by_id(id: str, k: int, dataset: str, model: str):
    # Load data from the database
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
        while not ids[idx].decode('utf-8').startswith(video_id) and idx < len(ids):
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
        while ids[idx + cur_i].decode('utf-8').startswith(video_id) and idx + cur_i < len(ids):
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

    # Return a list of (ID, feature, label, time) tuples
    return video_images


def get_random_video_frame(dataset: str, model: str):
    # Load data from the database
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

    # Return a list of (ID, feature, label, time) tuples
    return video_images


def get_images_by_temporal_query(query: str, query2: str, position: str, position2: str, k: int, dataset: str, model: str, is_life_log: bool):
    queries = [query, query2]
    separator = db.name_splitter(dataset).encode()

    # Tokenize query input and encode the data using the selected model
    text_features = [m.embed_text(q, model) for q in queries]

    # Normalize vector to make it smaller and for cosine calculcation
    text_features = [text_feature / text_feature.norm(dim=-1, keepdim=True) for text_feature in text_features]
    text_features = torch.stack(text_features)

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()

    if is_life_log:
        separator_ids = np.array([id[:11] for id in ids])
        splits = np.where(separator_ids[:-1] != separator_ids[1:])[0] + 1
        split_points = np.r_[0, splits, len(separator_ids) - 1]

        max_values_and_indices = []
        for text_f in text_features:
            _, sim = get_cosine_ranking(text_f, data)
            max_values_and_indices.append([
                (sim[split_points[i]:split_points[i+1]].max(), sim[split_points[i]:split_points[i+1]].argmax() + split_points[i])
                for i in range(len(split_points) - 1)
            ])

        day_splits = [i + 1 for i in range(len(split_points) - 2) if separator_ids[split_points[i]][:8] != separator_ids[split_points[i + 1]][:8]]
        days = np.split(split_points, day_splits)
        day_splits = np.r_[0, day_splits]

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

        sorted_indices = max_images[:k]
        similarities = [s.item() for seq in sorted_indices for s, _ in seq]
        sorted_indices = [idx for seq in sorted_indices for _, idx in seq]
    else:
        split_points = db.get_splits()
        num_segments = len(split_points) - 1

        # Precompute segment IDs
        segment_ids = torch.zeros(data.shape[0], dtype=torch.long)
        for i in range(num_segments):
            segment_ids[split_points[i]:split_points[i + 1]] = i
        segment_ids = segment_ids.to(data.device)

        num_segments = segment_ids.max().item() + 1

        scores = []
        argmax = []

        for i in range(text_features.shape[0]):
            sim_i = torch.matmul(text_features[i], data.T)  # shape: (N,)
            score_i, argmax_i = scatter_max(sim_i, segment_ids, dim=0, dim_size=num_segments)
            scores.append(score_i)
            argmax.append(argmax_i)

        scores = torch.stack(scores)    # shape: (T, S)
        argmax = torch.stack(argmax)    # shape: (T, S)

        # Aggregate across queries
        agg_scores = scores.sum(0)  # (S,)
        topk_scores, topk_seg = torch.topk(agg_scores, k)  # (k,)
        topk_idxs = argmax[:, topk_seg]  # (T, k)
        topk_vals = scores[:, topk_seg]  # (T, k)

        # Optional: convert to flat structure if needed
        sorted_indices = (topk_idxs.T.flatten() % data.shape[0])  # (T * k,)
        similarities = topk_vals.T.flatten()    # (T * k,)


    # Add a new value between each pair of consecutive indices
    extended_indices = []
    extended_similarities = []
    for i in range(0, len(sorted_indices) - 1, 2):
        i1 = sorted_indices[i]
        i2 = sorted_indices[i + 1]
        extended_indices.extend([i1.item(), (i1 + i2) // 2, i2.item(), i2 + 1 if i2 > i1 else i2 - 1])
        extended_similarities.extend([
            similarities[i].item(),
            0,
            similarities[i + 1].item(),
            0
        ])

    # If settings multiply and change to integer
    extended_indices = torch.tensor(extended_indices)
    selected_data = data[extended_indices]

    if c.BASE_MULTIPLICATION:
        selected_data = (selected_data * c.BASE_MULTIPLIER).int()

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, extended_indices, ids, dataset)

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[extended_indices].tolist(),
            list(range(len(extended_indices))),
            extended_similarities,
            selected_data.tolist(),
            labels[extended_indices].tolist(),
            db_time.tolist(),
        )
    )

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
            filter_indices = metadata[metadata[column] == value].index.tolist()
        elif column == "id" and value.startswith("yyyymm"):
            value = value[6:]
            filter_indices = metadata[metadata[column].str.slice(6,8) == value].index.tolist()
        elif column == "id" and value.startswith("yyyy"):
            value1 = value.replace("yyyy", "2019")
            filter_indices1 = metadata[metadata[column].str.startswith(value1)].index.tolist()
            value2 = value.replace("yyyy", "2020")
            filter_indices2 = metadata[metadata[column].str.startswith(value2)].index.tolist()
            filter_indices = filter_indices1 + filter_indices2
            indices = list(set(indices) & set(filter_indices))
            continue
        elif column == "hour" and "-" in value:
            value1, value2 = value.split("-")
            #l.logger.info(value1)
            filter_indices = []
            for i in range(int(value1), int(value2)):
                #l.logger.info(i)
                filter_indices += metadata[metadata[column].str.startswith(str(i).zfill(2))].index.tolist()
            indices = list(set(indices) & set(filter_indices))
            continue
        elif column == 'id':
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

    return most_similar_samples


def get_filters(dataset: str):
    # Load metadata from the database
    db.load_features(dataset, 'clip-vit-webli')
    metadata = db.get_metadata()

    # Get the list of available filters
    filters = metadata.columns.tolist()

    # Return the list of available filters
    return filters


def position_to_local_shortcut(position):
    positions = {"top-left": "tl", "top-right": "tr", "middle": "mi", "bottom-left": "bl", "bottom-right": "br"}
    return positions.get(position, -1)


def get_images_by_local_text_query(query: str, position: str, k: int, dataset: str, model: str, selected_indices: list):
    position = position_to_local_shortcut(position)

    # Tokenize query input and encode the data using the selected model
    text_features = m.embed_text(query, model)

    # Normalize vector to make it smaller and for cosine calculcation
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Load data
    db.load_features(dataset, model)
    data = db.get_local_data()
    if position in data:
        data = data[position]
    else:
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

    # Return a list of (ID, rank, score, feature, label, time) tuples
    return most_similar_samples


def decode_position_string(position_string: str):
    if position_string == "everywhere":
        return ["tl", "tm","tr","ml","mm","mr","bl", "bm","br"]
    else:
        return position_string.split("-")



def get_images_by_local_texture_query(query: str, position_source: str , position_target: str, k: int, dataset: str, model: str, selected_indices: list):
    positions_target = decode_position_string(position_target)

    # Load data
    db.load_features(dataset, model)
    data = db.get_texture_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()
    l.logger.info("DATA KEYS")
    l.logger.info(data)
    l.logger.info(data.keys())

    video_id, frame_id = db.uri_spliter(query, dataset)
    id = video_id + db.name_splitter(dataset) + frame_id
    idx = np.where(ids == id.encode('utf-8'))[0][0]

    input_feature = None
    data_input = None
    if position_source in data:
        data_input = data[position_source]
        input_feature = data_input[idx]
        l.logger.info("Position was used")

    if selected_indices is not None:
        if idx not in selected_indices:
            selected_indices.append(idx)
        data_input = data_input[selected_indices]
        ids = ids[selected_indices]
        labels = labels[selected_indices]
        if db_time.shape[0] > 0:
            db_time = db_time[selected_indices]


    list_sorted_indices = []
    list_similarities = []
    for pos in positions_target:
         data_output = data[pos]
         sorted_indices, similarities = get_texture_distance(input_feature, data_output, top_k=k)
         list_sorted_indices = list_sorted_indices + list(sorted_indices.tolist())
         list_similarities = list_similarities + list(similarities.tolist())

    # Zip the lists together and sort by list_similarities
    sorted_pairs = sorted(zip(list_similarities, list_sorted_indices), key=lambda x: x[0])

    list_similarities_sorted, list_sorted_indices_updated = zip(*sorted_pairs)

    # Convert back to lists (optional, as zip() returns tuples)
    unique_pairs = {}
    for similarity, index in zip(list_similarities_sorted, list_sorted_indices_updated):
        if index not in unique_pairs:
            unique_pairs[index] = similarity

    # Extract the results back into lists
    list_sorted_indices_unique = list(unique_pairs.keys())
    list_similarities_unique = list(unique_pairs.values())

    sorted_indices = list_sorted_indices_unique[:k]
    list_similarities_unique = list_similarities_unique[:k]

    # If settings multiply and change to integer
    selected_data = data_input[sorted_indices]
    if c.BASE_MULTIPLICATION:
        selected_data = (selected_data * c.BASE_MULTIPLIER).int()

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, sorted_indices, ids, dataset)

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            list_similarities_unique,
            selected_data.tolist(),
            labels[sorted_indices].tolist(),
            db_time.tolist(),
        )
    )

    # Return a list of (ID, rank, score, feature, label, time) tuples
    return most_similar_samples


def bayes_update(selected_images, max_rank, sub_ids, scores, dataset: str, model: str, alpha=0.01):
    sub_ids = np.array(['_'.join(pair) for pair in sub_ids])
    selected_images = set(['_'.join(pair) for pair in selected_images])

    # Load data
    db.load_features(dataset, model)
    data = db.get_data()
    ids = db.get_ids()
    labels = db.get_labels()
    db_time = db.get_time()
    scores = np.array(scores)

    str_ids = db.get_str_ids()

    # Filter to sub_ids
    sub_mask = np.isin(str_ids, sub_ids)
    if not sub_mask.any():
        return []

    data = data[sub_mask]
    ids = ids[sub_mask]
    labels = labels[sub_mask]
    db_time = db_time[sub_mask]

    str_ids = np.array([x.decode() if isinstance(x, bytes) else x for x in ids])
    scores_indexes = [int(np.where(str_ids == item)[0][0]) for item in sub_ids]
    scores = scores[scores_indexes]

    max_rank = min(len(ids), max_rank)

    # Positive and negative masks
    pos_mask = np.isin(str_ids, list(selected_images))
    neg_mask = np.zeros_like(pos_mask)
    neg_mask[:max_rank] = ~pos_mask[:max_rank]

    if not pos_mask.any() or not neg_mask.any():
        return []

    # Calculate products
    prod_positive = fast_matmul(data[pos_mask], data.T)
    prod_negative = fast_matmul(data[neg_mask], data.T)

    # Calculate PF and NF
    PF = torch.sum(torch.exp(- (1 - prod_positive) / alpha), dim=0)
    NF = torch.sum(torch.exp(- (1 - prod_negative) / alpha), dim=0) + 1e-8

    # Update scores with new weights
    updated_scores = scores * (PF / NF).cpu().numpy()

    sorted_indices = np.argsort(updated_scores)[::-1].copy()  # This removes negative strides

    # Get the time stamps for the sliced IDs
    db_time = get_time_stamps(db_time, sorted_indices, ids, dataset)

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            updated_scores[sorted_indices].tolist(),
            data[sorted_indices].tolist(),
            labels[sorted_indices].tolist(),
            db_time.tolist(),
        )
    )

    # Return a list of (ID, rank, score, feature, label, time) tuples
    return most_similar_samples
