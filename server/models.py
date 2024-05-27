# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time

import open_clip

from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

import logger as l
import configs as c


# List available models
available_models = ['clip-laion', 'clip-openai', 'clip-vit-webli']


# Store in memory as long as possible
cur_model_sel = ''
cur_model = None


# Load functions for the open clip LAION model
def load_laion():
    global cur_model_sel
    global cur_model

    if cur_model_sel != 'clip-laion':
        # Load model and tokenizer via open clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        )
        tokenizer = open_clip.get_tokenizer(
            'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        )

        cur_model_sel = 'clip-laion'
        cur_model = (model, tokenizer, preprocess)

    else:
        model, tokenizer, preprocess = cur_model

    return model, tokenizer, preprocess


# Embed text using LAION
def embed_text_laion(text):
    model, tokenizer, _ = load_laion()

    query_tokens = tokenizer(text)
    # text_features = model.encode_text(query_tokens).detach().cpu().numpy().flatten()
    text_features = model.encode_text(query_tokens).detach().cpu().flatten()

    return text_features


# Embed image using LAION
def embed_image_laion(image):
    model, _, preprocess = load_laion()

    query_image = preprocess(image).unsqueeze(0)
    # image_features = model.encode_image(query_image).detach().cpu().numpy().flatten()
    image_features = model.encode_image(query_image).detach().cpu().flatten()

    return image_features


# Load functions for the Open CLIP model
def load_open_clip():
    global cur_model_sel
    global cur_model

    if cur_model_sel != 'clip-openai':
        # Load model and tokenizer via transformers
        processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
        model = AutoModelForZeroShotImageClassification.from_pretrained(
            'openai/clip-vit-large-patch14'
        )

        cur_model_sel = 'clip-openai'
        cur_model = (model, processor)

    else:
        model, processor = cur_model

    return model, processor


# Embed text using Open CLIP
def embed_text_open_clip(text):
    model, processor = load_open_clip()

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)

    return outputs.text_embeds


# Embed image using Open CLIP
def embed_image_open_clip(image):
    model, processor = load_open_clip()

    inputs = processor(
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)

    return outputs.image_embeds


def load_vit_webli():
    global cur_model_sel
    global cur_model

    if cur_model_sel != 'clip-vit-webli':
        # Load model and tokenizer via open clip
        model, _, preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:timm/ViT-L-16-SigLIP-384'
        )
        tokenizer = open_clip.get_tokenizer(
            'hf-hub:timm/ViT-L-16-SigLIP-384'
        )

        cur_model_sel = 'clip-vit-webli'
        cur_model = (model, tokenizer, preprocess)

    else:
        model, processor = cur_model

    return model, processor


# Embed text using webli
def embed_text_laion(text):
    model, tokenizer, _ = load_vit_webli()

    query_tokens = tokenizer(text)
    text_features = model.encode_text(query_tokens).detach().cpu().flatten()

    return text_features


# Embed image using webli
def embed_image_laion(image):
    model, _, preprocess = load_vit_webli()

    query_image = preprocess(image).unsqueeze(0)
    image_features = model.encode_image(query_image).detach().cpu().flatten()

    return image_features



def embed_text(text, model):
    # Check if model is available
    if not model in available_models:
        model = available_models[0]

    # Check if the specified model is 'clip-laion'
    if 'clip-laion' in model:
        return embed_text_laion(text)
    # If the model is 'clip-openai', call the function for OpenAI's CLIP model
    elif 'clip-openai' in model:
        return embed_text_open_clip(text)


def embed_image(image, model):
    # Check if model is available
    if not model in available_models:
        model = available_models[0]

    # Check if the specified model is 'clip-laion'
    if 'clip-laion' in model:
        return embed_image_laion(image)
    # If the model is 'clip-openai', call the function for OpenAI's CLIP model
    elif 'clip-openai' in model:
        return embed_image_open_clip(image)


l.logger.info('Start to load pre-trained models')
start_time = time.time()
load_laion()
execution_time = time.time() - start_time
l.logger.info(f'Loading pre-trained models took: {execution_time:.6f} secs')
