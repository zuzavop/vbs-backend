# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import json
import logging

from PIL import Image
from io import BytesIO

from fastapi import FastAPI, File, Query, Body, UploadFile, Form, Response
from fastapi.middleware.cors import CORSMiddleware

import logger as l
import configs as c
import features as fs
import database as db


# Create an instance of the FastAPI class
app = FastAPI()
origins = ['*']  # ['http://localhost', 'http://acheron.ms.mff.cuni.cz/']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load features into the database
db.load_features()


class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format(x, f'.{c.BASE_ROUNDING_PRECISION}f'))


json.encoder.c_make_encoder = None
json.encoder.float = RoundingFloat


def attachment_response_creator(dict_for_json: dict) -> Response:
    start_time = time.time()
    dict_for_json_str = json.dumps(dict_for_json)
    headers = {'Content-Disposition': 'attachment; filename="data.json"'}
    resp = Response(dict_for_json_str, headers=headers, media_type='application/json')
    execution_time = time.time() - start_time
    l.logger.info(f'Attachment response creation: {execution_time:.6f} secs')
    return resp


def json_response_creator(dict_for_json: dict) -> Response:
    start_time = time.time()
    resp = Response(json.dumps(dict_for_json))
    execution_time = time.time() - start_time
    l.logger.info(f'Response creation: {execution_time:.6f} secs')
    return resp


def generate_min_return_dictionary(images, dataset, add_features=True, max_labels=10):
    # Create return dictionary
    ret_dict = []
    for ids, features, labels, time in images:
        if not add_features:
            features = []

        ids = ids.decode('utf-8')
        video_id, frame_id = ids.split(db.name_splitter(dataset), 1)
        ids = ids.replace('-', '_')

        if isinstance(labels, list):
            labels = labels[:max_labels]

        tmp_dict = {
            'uri': f'{dataset}/{video_id}/{ids}.jpg',
            'id': [video_id, frame_id],
            'features': features,
            'label': labels,
            'time': time,
        }
        ret_dict.append(tmp_dict)
    return ret_dict


def generate_return_dictionary(images, dataset, add_features=True, max_labels=10):
    # Create return dictionary
    ret_dict = []
    for ids, rank, score, features, labels, time in images:
        if not add_features:
            features = []

        ids = ids.decode('utf-8')
        video_id, frame_id = ids.split(db.name_splitter(dataset), 1)
        ids = ids.replace('-', '_')

        if isinstance(labels, list):
            labels = labels[:max_labels]

        tmp_dict = {
            'uri': f'{dataset}/{video_id}/{ids}.jpg',
            'rank': rank,
            'score': score,
            'id': [video_id, frame_id],
            'features': features,
            'label': labels,
            'time': time,
        }
        ret_dict.append(tmp_dict)
    return ret_dict


# Define the 'textQuery' route
@app.post('/textQuery/')
async def text_query(query_params: dict):
    '''
    Get a list of images based on a text query.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    query = query_params.get('query', '')
    k = min(query_params.get('k', c.BASE_K), 10000)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))

    # Call the function to retrieve images
    images = fs.get_images_by_text_query(query, k, dataset, model)

    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels)

    execution_time = time.time() - start_time
    l.logger.info(f'/textQuery: {execution_time:.6f} secs')

    if download_speed_up:
        return attachment_response_creator(ret_dict)
    else:
        return json_response_creator(ret_dict)


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

    k = min(query_params.get('k', 1000), 10000)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))

    try:
        # Read the uploaded image file
        image_data = await image.read()

        # Open the image using Pillow (PIL)
        uploaded_image = Image.open(BytesIO(image_data))

        images = fs.get_images_by_image_query(uploaded_image, k, dataset, model)

        # Create return dictionary
        ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels)

    except Exception as e:
        ret_dict = {'error': str(e)}

    execution_time = time.time() - start_time
    l.logger.info(f'/imageQuery: {execution_time:.6f} secs')

    if download_speed_up:
        return attachment_response_creator(ret_dict)
    else:
        return json_response_creator(ret_dict)


# Define the 'imageQueryByID' route
@app.post('/imageQueryByID/')
async def image_query_by_id(query_params: dict):
    '''
    Get a list of images based on an image query.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    video_id = query_params.get('video_id', '')
    frame_id = query_params.get('frame_id', '')
    item_id = query_params.get('item_id', '')
    k = min(query_params.get('k', 1000), 10000)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))

    # Call the function to retrieve images
    if item_id == '':
        item_id = f'{video_id}_{frame_id}'
    images = fs.get_images_by_image_id(item_id, k, dataset, model)

    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels)

    execution_time = time.time() - start_time
    l.logger.info(f'/imageQueryByID: {execution_time:.6f} secs')

    if download_speed_up:
        return attachment_response_creator(ret_dict)
    else:
        return json_response_creator(ret_dict)


# Define the 'temporalQuery' route
@app.post('/temporalQuery/')
async def text_query(query_params: dict):
    '''
    Get a list of images based on a temporal query.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    query = query_params.get('query', '')
    k = min(query_params.get('k', c.BASE_K), 10000)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))
    is_life_logging = bool(query_params.get('life_log', c.BASE_LIFE_LOG))

    # Call the function to retrieve images
    images = fs.get_images_by_temporal_query(query, k, dataset, model, is_life_logging)

    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels)

    execution_time = time.time() - start_time
    l.logger.info(f'/temporalQuery: {execution_time:.6f} secs')

    if download_speed_up:
        return attachment_response_creator(ret_dict)
    else:
        return json_response_creator(ret_dict)


@app.post('/filter/')
async def filter(query_params: dict):
    '''
    Get a list of images based on a filter.
    '''
    start_time = time.time()
    l.logger.info(query_params)
    
    filter = query_params.get('filter', '')
    filter_type = query_params.get('filter_type', 'tag')
    k = min(query_params.get('k', c.BASE_K), 10000)
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    max_labels = query_params.get('max_labels', c.BASE_MAX_LABELS)
    add_features = bool(query_params.get('add_features', c.BASE_ADD_FEATURES))
    download_speed_up = bool(query_params.get('speed_up', c.BASE_DOWNLOADING_SPEED_UP))
    
    # Call the function to retrieve images
    images = fs.filter_metadata(filter, filter_type, k, dataset)
    
    # Create return dictionary
    ret_dict = generate_return_dictionary(images, dataset, add_features, max_labels)
    
    execution_time = time.time() - start_time
    l.logger.info(f'/filter: {execution_time:.6f} secs')
    
    if download_speed_up:
        return attachment_response_creator(ret_dict)
    else:
        return json_response_creator(ret_dict)
    

# Define the 'getVideoFrames' route
@app.post('/getVideoFrames/')
async def get_video_frames(query_params: dict):
    '''
    Get a list of video images based on a video ID.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    video_id = query_params.get('video_id', '')
    frame_id = query_params.get('frame_id', '')
    item_id = query_params.get('item_id', '')
    k = min(query_params.get('k', 1000), 10000)
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
    ret_dict = generate_min_return_dictionary(images, dataset, add_features, max_labels)

    execution_time = time.time() - start_time
    l.logger.info(f'/getVideoFrames: {execution_time:.6f} secs')

    if download_speed_up:
        return attachment_response_creator(ret_dict)
    else:
        return json_response_creator(ret_dict)


# Define the 'getVideo' route
@app.get('/getVideo')
async def get_video(query_params: dict):
    '''
    Get URI of video.
    '''
    start_time = time.time()
    l.logger.info(f'Get video {video_id}')
    
    video_id = query_params.get('video_id', '')
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()

    # Call the function to retrieve video images
    video = fs.get_video_by_id(video_id, dataset)

    execution_time = time.time() - start_time
    l.logger.info(f'/getVideo: {execution_time:.6f} secs')

    return {'video_id': video_id, 'video': video}


@app.get('/getFiltres/')
async def get_filtres(query_params: dict):
    '''
    Get a list of filters.
    '''
    start_time = time.time()
    l.logger.info(query_params)
    
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    
    # Call the function to retrieve filters
    filters = fs.get_filters(dataset)
    
    execution_time = time.time() - start_time
    l.logger.info(f'/getFiltres: {execution_time:.6f} secs')
    
    return {'filters': filters}
    

# Define the 'getRandomFrame' route
@app.get('/getRandomFrame/')
async def get_random_frame(query_params: dict = {}):
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

    if download_speed_up:
        return attachment_response_creator(ret_dict)
    else:
        return json_response_creator(ret_dict)


# Define the 'getTimeInterval' route
@app.get('/getTimeInterval/')
async def get_time_interval(query_params: dict = {}):
    '''
    Get the time interval for a video frame.
    '''
    dataset = query_params.get('dataset', c.BASE_DATASET).upper()
    model = query_params.get('model', c.BASE_MODEL)


# Define a route and its handler function
@app.get('/')
async def read_root():
    return {'message': 'Server is running!'}


# Run the FastAPI application
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
    
    

