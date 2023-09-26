import h5py
import time
import numpy as np
import open_clip

from numba import jit
from PIL import Image

from scipy.spatial.distance import cosine

import database as db
import logger as l
import model as m


def get_cosine_ranking(query_vector, matrix):
    # get the dot product for every entry
    dot_product = np.matmul(query_vector, matrix.T)
    # sort for the indices of the nearest neighbors
    nearest_neighbors = np.argsort(-dot_product)
    return nearest_neighbors, dot_product


def get_images_by_text_query(query: str, k: int):
    start_time = time.time()

    # load model depending on selcted model
    model = m.loaded_model
    tokenizer = m.loaded_tokenizer

    # tokenize query input and encode the data using the selected model
    query_tokens = tokenizer(query)
    text_features = model.encode_text(query_tokens).detach().cpu().numpy().flatten()

    # normalize vector to make it smaller and for cosine calculcation
    text_features = text_features / np.linalg.norm(text_features)

    # load data
    data = db.get_data()
    ids = np.array(db.get_ids())

    # calculate cosine distance between embedding and data and sort distances
    sorted_indices, distances = get_cosine_ranking(text_features, data)
    sorted_indices = sorted_indices[:k]
    distances = distances[sorted_indices]

    # give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            data[sorted_indices].tolist(),
            distances.tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del distances
    del sorted_indices

    return most_similar_samples


def get_images_by_image_query(image: Image, k: int):
    start_time = time.time()

    # load model depending on selcted model
    model = m.loaded_model
    preprocess = m.loaded_preprocess

    # preprocess query input and encode the data using the selected model
    query_image = preprocess(image).unsqueeze(0)
    image_features = model.encode_image(query_image).detach().cpu().numpy().flatten()

    # normalize vector to make it smaller and for cosine calculcation
    image_features = image_features / np.linalg.norm(image_features)

    # load data
    data = db.get_data()
    ids = np.array(db.get_ids())

    # calculate cosine distance between embedding and data and sort distances
    sorted_indices, distances = get_cosine_ranking(image_features, data)
    sorted_indices = sorted_indices[:k]
    distances = distances[sorted_indices]

    # give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            ids[sorted_indices].tolist(),
            data[sorted_indices].tolist(),
            distances.tolist(),
        )
    )

    execution_time = time.time() - start_time
    l.logger.info(f'Getting nearest embeddings: {execution_time:.6f} secs')

    del data
    del ids
    del distances
    del sorted_indices

    return most_similar_samples


# Dummy function to simulate fetching video images based on a video ID
def get_video_images_by_id(video_id: str, k: int):
    # In a real application, you would fetch video images based on the video ID.
    # Here, we're returning dummy data for demonstration purposes.
    video_images = [f'Image {i} for Video ID {video_id}' for i in range(1, k + 1)]
    return video_images


# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-large-patch14")
