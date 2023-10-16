import h5py
import time
import numpy as np
import open_clip

from PIL import Image

from scipy.spatial.distance import cosine

import database as db
import logger as l
import model as m


def get_cosine_ranking(query_vector, matrix):
    # Get the dot product for every entry
    dot_product = np.matmul(query_vector, matrix.T)

    # Sort for the indices of the nearest neighbors
    nearest_neighbors = np.argsort(-dot_product)
    return nearest_neighbors, dot_product


def get_images_by_text_query(query: str, k: int):
    start_time = time.time()

    # Load model depending on selcted model
    model = m.loaded_model
    tokenizer = m.loaded_tokenizer

    # Tokenize query input and encode the data using the selected model
    query_tokens = tokenizer(query)
    text_features = model.encode_text(query_tokens).detach().cpu().numpy().flatten()

    # Normalize vector to make it smaller and for cosine calculcation
    text_features = text_features / np.linalg.norm(text_features)

    # Load data
    data = db.get_data()
    ids = np.array(db.get_ids())

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(text_features, data)
    sorted_indices = sorted_indices[:k]
    similarities = similarities[sorted_indices]

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities.tolist(),
            data[sorted_indices].tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del similarities
    del sorted_indices

    return most_similar_samples


def get_images_by_image_query(image: Image, k: int):
    start_time = time.time()

    # Load model depending on selcted model
    model = m.loaded_model
    preprocess = m.loaded_preprocess

    # Preprocess query input and encode the data using the selected model
    query_image = preprocess(image).unsqueeze(0)
    image_features = model.encode_image(query_image).detach().cpu().numpy().flatten()

    # Normalize vector to make it smaller and for cosine calculcation
    image_features = image_features / np.linalg.norm(image_features)

    # Load data
    data = db.get_data()
    ids = np.array(db.get_ids())

    # Calculate cosine distance between embedding and data and sort similarities
    sorted_indices, similarities = get_cosine_ranking(image_features, data)
    sorted_indices = sorted_indices[:k]
    similarities = similarities[sorted_indices]

    # Give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            [i for i in range(len(sorted_indices))],
            similarities.tolist(),
            data[sorted_indices].tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del similarities
    del sorted_indices

    return most_similar_samples


def get_video_images_by_id(id: str, k: int):
    # Load data from the database
    # Get an array of video IDs from the database
    data = db.get_data()
    ids = np.array(db.get_ids())

    # Find the index of the provided 'id' within the 'ids' array
    idx = np.where(ids == id.encode('utf-8'))[0][0]

    # Extract a slice of 'k' elements centered around the found index
    ids = ids[idx - k : idx + k]
    features = data[idx - k : idx + k]

    # Combine the selected IDs and features into a list of tuples
    video_images = list(zip(ids.tolist(), features.tolist()))

    # Return a list of (ID, feature) pairs
    return video_images


def get_random_video_frame():
    # Load data from the database
    # Get an array of video IDs from the database
    data = db.get_data()
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


# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-large-patch14")
