import h5py
import numpy as np
import open_clip

from numba import jit
from PIL import Image

from scipy.spatial.distance import cosine

import database as db
import logger as l


def load_open_clip():
    # load model and tokenizer via open clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    )
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    return model, tokenizer, preprocess


@jit(nopython=True)
def cosine_similarity_numba(u: np.ndarray, v: np.ndarray):
    assert u.shape[0] == v.shape[0]
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / np.sqrt(uu * vv)
    return cos_theta


def get_images_by_text_query(query: str, k: int):
    # load model depending on selcted model
    model, tokenizer, _ = load_open_clip()

    # tokenize query input and encode the data using the selected model
    query_tokens = tokenizer(query)
    text_features = model.encode_text(query_tokens).detach().cpu().numpy().flatten()

    # load data
    data = db.get_data()
    idx = np.array(db.get_idx())

    # calculate cosine distance between embedding and data and sort distances
    distances = np.apply_along_axis(
        lambda x: cosine_similarity_numba(text_features, x), axis=1, arr=data
    )
    sorted_indices = np.argsort(distances)[::-1]

    # give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            idx[sorted_indices[:k]].tolist(),
            data[sorted_indices][:k].tolist(),
            distances[sorted_indices][:k].tolist(),
        )
    )
    return most_similar_samples


def get_images_by_image_query(image: Image, k: int):
    # load model depending on selcted model
    model, _, preprocess = load_open_clip()

    # preprocess query input and encode the data using the selected model
    query_image = preprocess(image).unsqueeze(0)
    image_features = model.encode_image(query_image).detach().cpu().numpy().flatten()

    # load data
    data = db.get_data()
    idx = np.array(db.get_idx())

    # calculate cosine distance between embedding and data and sort distances
    distances = np.apply_along_axis(
        lambda x: cosine_similarity_numba(image_features, x), axis=1, arr=data
    )
    sorted_indices = np.argsort(distances)[::-1]

    # give only back the k most similar embeddings
    most_similar_samples = list(
        zip(
            idx[sorted_indices[:k]].tolist(),
            data[sorted_indices][:k].tolist(),
            distances[sorted_indices][:k].tolist(),
        )
    )
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
