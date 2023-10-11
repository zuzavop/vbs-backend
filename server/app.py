from fastapi import FastAPI, File, Query, Body, UploadFile

import logging

from PIL import Image
from io import BytesIO

import features as fs
import database as db
import logger as l


# Create an instance of the FastAPI class
app = FastAPI()

# Load features into the database
db.load_features()


# Define the 'textQuery' route
@app.post('/textQuery/')
async def text_query(query_params: dict, get_embeddings: bool = False):
    '''
    Get a list of images based on a text query.
    '''
    l.logger.info(query_params)
    query = query_params.get('query', '')
    k = min(query_params.get('k', 100), 10000)
    dataset = query_params.get('dataset', '')
    model = query_params.get('model', '')

    # Call the function to retrieve images
    images = fs.get_images_by_text_query(query, k)

    # Create return dictionary
    ret_dict = []
    for ids, rank, score, features in images:
        if not get_embeddings:
            features = []

        video_id, frame_id = ids.decode('utf-8').split('_', 1)

        tmp_dict = {
            'uri': f'{video_id}/{ids}.jpg',
            'rank': rank,
            'score': score,
            'id': [video_id, frame_id],
            'features': features,
            'label': None,
        }
        ret_dict.append(tmp_dict)

    return ret_dict


# Define the 'imageQuery' route
@app.post('/imageQuery/')
async def image_query(
    image: UploadFile,
    k: int = 1000,
    get_embeddings: bool = False,
):
    '''
    Get a list of images based on an image query.
    '''
    # Call the function to retrieve images
    try:
        # Read the uploaded image file
        image_data = await image.read()

        # Open the image using Pillow (PIL)
        uploaded_image = Image.open(BytesIO(image_data))
        k = min(k, 10000)

        images = fs.get_images_by_image_query(uploaded_image, k)

        # Create return dictionary
        ret_dict = []
        for ids, rank, score, features in images:
            if not get_embeddings:
                features = []

            video_id, frame_id = ids.decode('utf-8').split('_', 1)

            tmp_dict = {
                'uri': f'{video_id}/{ids}.jpg',
                'rank': rank,
                'score': score,
                'id': [video_id, frame_id],
                'features': features,
                'label': None,
            }
            ret_dict.append(tmp_dict)

        return ret_dict

    except Exception as e:
        return {"error": str(e)}


# Define the 'imageQueryByID' route
@app.post('/imageQueryByID/')
async def image_query_by_id(
    video_id: str,
    frame_id: str,
    k: int = 1000,
    get_embeddings: bool = False,
):
    '''
    Get a list of images based on an image query.
    '''
    # Call the function to retrieve images
    try:
        # Read the uploaded image file
        image_data = await image.read()

        # Open the image using Pillow (PIL)
        uploaded_image = Image.open(BytesIO(image_data))
        k = min(k, 10000)

        images = fs.get_images_by_image_query(uploaded_image, k)

        # Create return dictionary
        ret_dict = []
        for ids, rank, score, features in images:
            if not get_embeddings:
                features = []

            video_id, frame_id = ids.decode('utf-8').split('_', 1)

            tmp_dict = {
                'uri': f'{video_id}/{ids}.jpg',
                'rank': rank,
                'score': score,
                'id': [video_id, frame_id],
                'features': features,
                'label': None,
            }
            ret_dict.append(tmp_dict)

        return ret_dict

    except Exception as e:
        return {"error": str(e)}


# Define the 'getVideoFrames' route
@app.get('/getVideoFrames')
async def get_video_frames(video_id: str, frame_id: str):
    '''
    Get a list of video images based on a video ID.
    '''
    # Call the function to retrieve video images
    video_image = fs.get_video_image_by_id(video_id, frame_id)

    return {'video_id': video_id, 'frame_id': frame_id, 'video_image': video_image}


# Define a route and its handler function
@app.get('/')
async def read_root():
    return {'message': 'Server is running!'}


# Run the FastAPI application
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
