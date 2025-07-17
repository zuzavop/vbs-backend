# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import json
import time
from io import BytesIO
from typing import Dict, Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import logger as l
import configs as c
import features as fs
import database as db


# Create an instance of the FastAPI class
app = FastAPI(
    title="VBS Backend API",
    description="Video Browsing System Backend API for multimedia retrieval",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# Initialize the application
def initialize_app():
    """Initialize the application with required data."""
    try:
        db.load_features(c.BASE_DATASET, c.BASE_MODEL, True)
        l.logger.info("Application initialized successfully")
    except Exception as e:
        l.logger.error(f"Failed to initialize application: {e}")
        raise


# Load features into the database on startup
initialize_app()


# Custom JSON encoder for float precision
class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format(x, f'.{c.BASE_ROUNDING_PRECISION}f'))


json.encoder.c_make_encoder = None
json.encoder.float = RoundingFloat


# Utility functions
def create_response(data: Dict[str, Any], speed_up: bool = False) -> Response:
    """Create a JSON response with timing information."""
    start_time = time.time()
    
    if speed_up:
        response = JSONResponse(data, media_type='application/json')
    else:
        response = Response(json.dumps(data))
    
    execution_time = time.time() - start_time
    l.logger.info(f'Response creation: {execution_time:.6f} secs')
    return response


def generate_min_return_dictionary(images, dataset, add_features=True, max_labels=10, start_time=0):
    # Create return dictionary
    ret_dict = []
    for ids, features, labels, timestamp in images:
        if not add_features:
            features = []

        ids = ids.decode('utf-8')
        video_id, frame_id = db.uri_spliter(ids, dataset)
        ids = ids.replace('-', '_')

        if isinstance(labels, list):
            labels = labels[:max_labels]

        tmp_dict = {
            'uri': f'{dataset}/{video_id}/{ids}.jpg',
            'id': [video_id, frame_id],
            'label': labels,
            'time': timestamp,
        }
        if add_features:
            tmp_dict['features'] = features
        ret_dict.append(tmp_dict)

    ret_dict.append({
        "T3_dataServiceReceived": start_time,
        "T4_dataServiceProcessedQuery": time.time(),
    })

    return ret_dict


def generate_return_dictionary(images, dataset, add_features=True, max_labels=10, start_time=0):
    # Create return dictionary
    ret_dict = []
    for ids, rank, score, features, labels, timestamp in images:
        if not add_features:
            features = []

        ids = ids.decode('utf-8')
        video_id, frame_id = db.uri_spliter(ids, dataset)
        ids = ids.replace('-', '_')

        if isinstance(labels, list):
            labels = labels[:max_labels]

        tmp_dict = {
            'uri': f'{dataset}/{video_id}/{ids}.jpg',
            'rank': np.array(rank).astype(str).tolist(),
            'score': np.array(score).astype(str).tolist(),
            'id': [video_id, frame_id],
            'label': np.array(labels).astype(str).tolist(),
            'time': timestamp,
        }
        if add_features:
            tmp_dict['features'] = np.array(features).astype(str).tolist()
        ret_dict.append(tmp_dict)

    ret_dict.append({
        "T3_dataServiceReceived": start_time,
        "T4_dataServiceProcessedQuery": time.time(),
    })

    return ret_dict


# Define the 'textQuery' route
@app.post('/textQuery/')
def text_query(query_params: dict):
    '''
    Get a list of images based on a text query.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    query = query_params.get('query', '')
    position = query_params.get('position', '')
    k = query_params.get('k', c.BASE_K)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))
    filter = query_params.get('filters', {})

    if filter == {}:
        selected_indeces = None
    else:
        selected_indeces = fs.get_filter_indices(filter, dataset)

    # Call the function to retrieve images
    if position == '':
        images = fs.get_images_by_text_query(query, k, dataset, model, selected_indeces)
    else:
        images = fs.get_images_by_local_text_query(query, position, k, dataset, model, selected_indeces)

    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels, start_time)

    execution_time = time.time() - start_time
    l.logger.info(f'/textQuery: {execution_time:.6f} secs')

    return create_response(ret_dict, download_speed_up)


# Define the 'imageQuery' route
@app.post('/imageQuery/')
async def image_query(
    image: UploadFile,
    query_params: str = Form(...),
):
    '''
    Get a list of images based on an image query.
    '''
    start_time = time.time()
    query_params = json.loads(query_params)
    l.logger.info(query_params)

    k = query_params.get('k', c.BASE_K)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))
    filter = query_params.get('filters', {})

    if filter == {}:
        selected_indeces = None
    else:
        selected_indeces = fs.get_filter_indices(filter, dataset)

    try:
        # Read the uploaded image file
        image_data = await image.read()

        # Open the image using Pillow (PIL)
        uploaded_image = Image.open(BytesIO(image_data))

        images = fs.get_images_by_image_query(uploaded_image, k, dataset, model, selected_indeces)

        # Create return dictionary
        ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels, start_time)

    except Exception as e:
        ret_dict = {'error': str(e)}

    execution_time = time.time() - start_time
    l.logger.info(f'/imageQuery: {execution_time:.6f} secs')

    return create_response(ret_dict, download_speed_up)


# Define the 'imageQueryByID' route
@app.post('/imageQueryByID/')
def image_query_by_id(query_params: dict):
    '''
    Get a list of images based on an image query.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    video_id = query_params.get('video_id', '')
    frame_id = query_params.get('frame_id', '')
    item_id = query_params.get('item_id', '')
    k = query_params.get('k', c.BASE_K)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))
    filter = query_params.get('filters', {})

    if filter == {}:
        selected_indeces = None
    else:
        selected_indeces = fs.get_filter_indices(filter, dataset)

    # Call the function to retrieve images
    if item_id == '':
        item_id = f'{video_id}_{frame_id}'
    images = fs.get_images_by_image_id(item_id, k, dataset, model, selected_indeces)

    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels, start_time)

    execution_time = time.time() - start_time
    l.logger.info(f'/imageQueryByID: {execution_time:.6f} secs')

    return create_response(ret_dict, download_speed_up)


# Define the 'temporalQuery' route
@app.post('/temporalQuery/')
def temporal_query(query_params: dict):
    '''
    Get a list of images based on a temporal query.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    query = query_params.get('query', '')
    query2 = query_params.get('query2', '')
    position = query_params.get('position', '')
    position2 = query_params.get('position_scene_2', '')
    k = query_params.get('k', c.BASE_K)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))
    is_life_logging = dataset == 'LSC'

    # Call the function to retrieve images
    images = fs.get_images_by_temporal_query(query, query2, position, position2, k, dataset, model, is_life_logging)

    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels, start_time)

    execution_time = time.time() - start_time
    l.logger.info(f'/temporalQuery: {execution_time:.6f} secs')

    return create_response(ret_dict, download_speed_up)


@app.post('/filter/')
def filter(query_params: dict):
    '''
    Get a list of images based on a filter.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    filter = query_params.get('filters', {})
    k = query_params.get('k', c.BASE_K)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))

    # Call the function to retrieve images
    images = fs.filter_metadata(filter, k, dataset)

    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels, start_time)

    execution_time = time.time() - start_time
    l.logger.info(f'/filter: {execution_time:.6f} secs')

    return create_response(ret_dict, download_speed_up)


# Define the 'getVideoFrames' route
@app.post('/getVideoFrames/')
def get_video_frames(query_params: dict):
    '''
    Get a list of video images based on a video ID.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    video_id = query_params.get('video_id', '')
    frame_id = query_params.get('frame_id', '')
    item_id = query_params.get('item_id', '')
    k = query_params.get('k', c.BASE_K)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))

    # Call the function to retrieve video images
    if item_id == '':
        item_id = f'{video_id}_{frame_id}'
    images = fs.get_video_images_by_id(item_id, k, dataset, model)

    # Create return dictionary
    ret_dict = generate_min_return_dictionary(images, dataset, add_features, max_labels, start_time)

    execution_time = time.time() - start_time
    l.logger.info(f'/getVideoFrames: {execution_time:.6f} secs')

    return create_response(ret_dict, download_speed_up)


@app.get('/getFilters/')
def get_filters(dataset: str):
    '''
    Get a list of filters.
    '''
    start_time = time.time()
    l.logger.info(dataset)

    dataset = dataset.upper()

    # Call the function to retrieve filters
    filters = fs.get_filters(dataset)

    execution_time = time.time() - start_time
    l.logger.info(f'/getFiltres: {execution_time:.6f} secs')

    return {'filters': filters}


# Define the 'getRandomFrame' route
@app.get('/getRandomFrame/')
def get_random_frame(query_params: dict = {}):
    '''
    Get URI of random frame.
    '''
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))

    # Call the function to retrieve a random video frame
    images = fs.get_random_video_frame(dataset, model)

    # Create return dictionary
    ret_dict = generate_min_return_dictionary(images, dataset, add_features, max_labels)

    return create_response(ret_dict, download_speed_up)


@app.post('/textureQuery/')
def texture_query(query_params: dict):
    '''
    Get a list of images based on a text query.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    video_id = query_params.get('video_id', '')
    frame_id = query_params.get('frame_id', '')
    item_id = query_params.get('item_id', '')
    position_source = query_params.get('position_source', '')
    position_target = query_params.get('position_target', '')
    k = query_params.get('k', c.BASE_K)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))
    filter = query_params.get('filters', {})

    if filter == {}:
        selected_indeces = None
    else:
        selected_indeces = fs.get_filter_indices(filter, dataset)

    # Call the function to retrieve images
    if item_id == '':
        item_id = f'{video_id}_{frame_id}'

    images = fs.get_images_by_local_texture_query(item_id, position_source, position_target, k, dataset, model, selected_indeces)

    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels)

    execution_time = time.time() - start_time
    l.logger.info(f'/textureQuery: {execution_time:.6f} secs')

    return create_response(ret_dict, download_speed_up)


# Define the 'bayes' route
@app.post('/bayes/')
def bayes(query_params: dict):
    '''
    Get a list of images based on bayes update.
    '''
    start_time = time.time()

    selected_images = query_params.get('selected_images', [])
    ids = query_params.get('ids', [])
    max_rank = query_params.get('max_rank', 0)
    scores = query_params.get('scores', [])
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))

    if max_rank == 0:
        max_rank = len(ids)

    # Call the function to retrieve images
    images = fs.bayes_update(selected_images, max_rank, ids, scores, dataset, model)

    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels, start_time)

    execution_time = time.time() - start_time
    l.logger.info(f'/bayes: {execution_time:.6f} secs')

    return create_response(ret_dict, download_speed_up)


@app.post('/returnOne/')
async def return_of_one(query_params: dict):
    """Return Response with value 1. Works for testing purposes."""
    return JSONResponse(content=1)


@app.get('/')
async def read_root():
    """Root endpoint - health check."""
    return {'message': 'VBS Backend Server is running!', 'status': 'healthy'}


# Application startup
if __name__ == '__main__':
    import uvicorn
    
    l.logger.info("Starting VBS Backend Server...")
    uvicorn.run(
        app, 
        host='0.0.0.0', 
        port=8000
    )


