import logger as l
import configs as c

import time
import json
import logging

from PIL import Image
from io import BytesIO

from fastapi import FastAPI, File, Query, Body, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

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


# Define the 'textQuery' route
@app.post('/textQuery/')
async def text_query(query_params: dict):
    '''
    Get a list of images based on a text query.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    query = query_params.get('query', '')
    k = min(query_params.get('k', 100), 10000)
    dataset = query_params.get('dataset', c.BASE_DATASET)
    model = query_params.get('model', c.BASE_MODEL)
    add_features = bool(query_params.get('add_features', 0))

    # Call the function to retrieve images
    images = fs.get_images_by_text_query(query, k, dataset, model)

    # Create return dictionary
    ret_dict = []
    for ids, rank, score, features, labels in images:
        if not add_features:
            features = []

        ids = ids.decode('utf-8')
        video_id, frame_id = ids.split('_', 1)

        tmp_dict = {
            'uri': f'{dataset}/{video_id}/{ids}.jpg',
            'rank': rank,
            'score': score,
            'id': [video_id, frame_id],
            'features': features,
            'label': labels,
        }
        ret_dict.append(tmp_dict)

    execution_time = time.time() - start_time
    l.logger.info(f'/textQuery: {execution_time:.6f} secs')

    return JSONResponse(content=jsonable_encoder(ret_dict))


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

    k = min(query_params.get('k', 100), 10000)
    dataset = query_params.get('dataset', c.BASE_DATASET)
    model = query_params.get('model', c.BASE_MODEL)
    add_features = bool(query_params.get('add_features', 0))

    try:
        # Read the uploaded image file
        image_data = await image.read()

        # Open the image using Pillow (PIL)
        uploaded_image = Image.open(BytesIO(image_data))

        images = fs.get_images_by_image_query(uploaded_image, k, dataset, model)

        # Create return dictionary
        ret_dict = []
        for ids, rank, score, features, labels in images:
            if not add_features:
                features = []

            ids = ids.decode('utf-8')
            video_id, frame_id = ids.split('_', 1)

            tmp_dict = {
                'uri': f'{dataset}/{video_id}/{ids}.jpg',
                'rank': rank,
                'score': score,
                'id': [video_id, frame_id],
                'features': features,
                'label': labels,
            }
            ret_dict.append(tmp_dict)

    except Exception as e:
        ret_dict = {'error': str(e)}

    execution_time = time.time() - start_time
    l.logger.info(f'/imageQuery: {execution_time:.6f} secs')

    return JSONResponse(content=jsonable_encoder(ret_dict))


# Define the 'imageQueryByID' route
@app.post('/imageQueryByID/')
async def image_query_by_id(
    query_params: str = dict,
):
    '''
    Get a list of images based on an image query.
    '''
    start_time = time.time()
    query_params = json.loads(query_params)
    l.logger.info(query_params)

    video_id = query_params.get('video_id', '')
    frame_id = query_params.get('frame_id', '')
    k = min(query_params.get('k', 100), 10000)
    dataset = query_params.get('dataset', c.BASE_DATASET)
    model = query_params.get('model', c.BASE_MODEL)
    add_features = bool(query_params.get('add_features', 0))

    # Call the function to retrieve images
    try:
        # Read the uploaded image file
        image_data = await image.read()

        # Open the image using Pillow (PIL)
        uploaded_image = Image.open(BytesIO(image_data))

        images = fs.get_images_by_image_query(uploaded_image, k, dataset, model)

        # Create return dictionary
        ret_dict = []
        for ids, rank, score, features, labels in images:
            if not get_embeddings:
                features = []

            ids = ids.decode('utf-8')
            video_id, frame_id = ids.split('_', 1)

            tmp_dict = {
                'uri': f'{dataset}/{video_id}/{ids}.jpg',
                'rank': rank,
                'score': score,
                'id': [video_id, frame_id],
                'features': features,
                'label': labels,
            }
            ret_dict.append(tmp_dict)

    except Exception as e:
        ret_dict = {'error': str(e)}

    execution_time = time.time() - start_time
    l.logger.info(f'/imageQueryByID: {execution_time:.6f} secs')

    return JSONResponse(content=jsonable_encoder(ret_dict))


# Define the 'getVideoFrames' route
@app.post('/getVideoFrames/')
async def get_video_frames(query_params: dict):
    '''
    Get a list of video images based on a video ID.
    '''
    start_time = time.time()
    l.logger.info(query_params)

    id = query_params.get('item_id', '')
    k = min(query_params.get('k', 100), 10000)
    dataset = query_params.get('dataset', c.BASE_DATASET)
    model = query_params.get('model', c.BASE_MODEL)
    add_features = bool(query_params.get('add_features', 0))

    # Call the function to retrieve video images
    images = fs.get_video_images_by_id(id, k, dataset, model)

    # Create return dictionary
    ret_dict = []
    for ids, features in images:
        if not add_features:
            features = []

        ids = ids.decode('utf-8')
        video_id, frame_id = ids.split('_', 1)

        tmp_dict = {
            'uri': f'{dataset}/{video_id}/{ids}.jpg',
            'id': [video_id, frame_id],
            'features': features,
            'label': None,
        }
        ret_dict.append(tmp_dict)

    execution_time = time.time() - start_time
    l.logger.info(f'/getVideoFrames: {execution_time:.6f} secs')

    return JSONResponse(content=jsonable_encoder(ret_dict))


# Define the 'getVideo' route
@app.get('/getVideo')
async def get_video(video_id: str):
    '''
    Get URI of video.
    '''
    start_time = time.time()
    l.logger.info(f'Get video {video_id}')

    # Call the function to retrieve video images
    video_image = fs.get_video_image_by_id(video_id, frame_id)

    execution_time = time.time() - start_time
    l.logger.info(f'/getVideo: {execution_time:.6f} secs')

    return {'video_id': video_id, 'frame_id': frame_id, 'video_image': video_image}


# Define the 'getRandomFrame' route
@app.get('/getRandomFrame/')
async def get_random_frame():
    '''
    Get URI of random frame.
    '''
    add_features = False

    # Call the function to retrieve a random video frame
    images = fs.get_random_video_frame()

    # Create return dictionary
    ret_dict = []
    for ids, features in images:
        if not add_features:
            features = []

        ids = ids.decode('utf-8')
        video_id, frame_id = ids.split('_', 1)

        tmp_dict = {
            'uri': f'{video_id}/{ids}.jpg',
            'id': [video_id, frame_id],
            'features': features,
            'label': None,
        }
        ret_dict.append(tmp_dict)

    return JSONResponse(content=jsonable_encoder(ret_dict))


# Define a route and its handler function
@app.get('/')
async def read_root():
    return {'message': 'Server is running!'}


# Run the FastAPI application
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
