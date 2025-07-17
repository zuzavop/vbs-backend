# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import queue
import threading
from contextlib import contextmanager

import open_clip
import torch

from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

import logger as l
import configs as c

import os

torch.set_num_threads(12)
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"


# List available models
available_models = ['clip-laion', 'clip-openai', 'clip-vit-webli', 'clip-vit-so400m']


# Store model pools and current selection
model_pools = {}
pool_lock = threading.Lock()


class ModelPool:
    """Thread-safe model pool that maintains multiple instances of a model."""
    
    def __init__(self, model_name, pool_size=None):
        self.model_name = model_name
        self.pool_size = pool_size or c.MODEL_POOL_SIZE
        self.pool = queue.Queue(maxsize=self.pool_size)
        self.initialized = False
        self.init_lock = threading.Lock()
    
    def _create_model_instance(self):
        """Create a single model instance based on model name."""
        if 'clip-laion' in self.model_name:
            return self._create_laion_instance()
        elif 'clip-vit-webli' in self.model_name:
            return self._create_webli_instance()
        elif 'clip-openai' in self.model_name:
            return self._create_openai_instance()
        elif 'clip-vit-so400m' in self.model_name:
            return self._create_so400m_instance()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _create_laion_instance(self):
        """Create LAION model instance."""
        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        )
        tokenizer = open_clip.get_tokenizer(
            'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        )
        return (model, tokenizer, preprocess)
    
    def _create_webli_instance(self):
        """Create WebLI model instance."""
        model, preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:timm/ViT-L-16-SigLIP-384'
        )
        tokenizer = open_clip.get_tokenizer(
            'hf-hub:timm/ViT-L-16-SigLIP-384'
        )
        return (model, tokenizer, preprocess)
    
    def _create_openai_instance(self):
        """Create OpenAI CLIP model instance."""
        processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14')
        model = AutoModelForZeroShotImageClassification.from_pretrained(
            'openai/clip-vit-large-patch14'
        )
        return (model, processor)
    
    def _create_so400m_instance(self):
        """Create SO400M model instance."""
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-SO400M-14-SigLIP-384',
            pretrained="webli")

        checkpoint_path = 'model/MCIP-ViT-SO400M-14-SigLIP-384.pth'
        mcip_state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(mcip_state_dict, strict=True)

        tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384')
        return (model, tokenizer, preprocess)
    
    def initialize_pool(self):
        """Initialize the model pool with instances."""
        if self.initialized:
            return
        
        with self.init_lock:
            if self.initialized:
                return
            
            l.logger.info(f"Initializing model pool for {self.model_name} with {self.pool_size} instances")
            start_time = time.time()
            
            for i in range(self.pool_size):
                l.logger.info(f"Creating model instance {i+1}/{self.pool_size} for {self.model_name}")
                model_instance = self._create_model_instance()
                self.pool.put(model_instance)
            
            self.initialized = True
            execution_time = time.time() - start_time
            l.logger.info(f"Model pool initialization for {self.model_name} took: {execution_time:.6f} secs")
    
    @contextmanager
    def get_model(self):
        """Get a model instance from the pool with automatic return."""
        if not self.initialized:
            self.initialize_pool()
        
        try:
            model_instance = self.pool.get()
            l.logger.debug(f"Retrieved model instance from pool (remaining: {self.pool.qsize()})")
            yield model_instance
        except queue.Empty:
            l.logger.error(f"Timeout waiting for model instance from pool")
            raise RuntimeError("Model pool timeout - all instances are busy")
        finally:
            self.pool.put(model_instance)
            l.logger.debug(f"Returned model instance to pool (available: {self.pool.qsize()})")


def get_model_pool(model_name):
    """Get or create a model pool for the specified model."""
    global model_pools
    
    with pool_lock:
        if model_name not in model_pools:
            l.logger.info(f"Creating new model pool for {model_name}")
            model_pools[model_name] = ModelPool(model_name)
        
        return model_pools[model_name]


# Load functions for the open clip LAION model
def load_laion():
    """Get LAION model pool - kept for backward compatibility."""
    pool = get_model_pool('clip-laion')
    return pool


# Embed text using LAION
def embed_text_laion(text):
    pool = get_model_pool('clip-laion')
    
    with pool.get_model() as (model, tokenizer, _):
        query_tokens = tokenizer(text)
        text_features = model.encode_text(query_tokens).detach().cpu().flatten()
        return text_features


# Embed image using LAION
def embed_image_laion(image):
    pool = get_model_pool('clip-laion')
    
    with pool.get_model() as (model, _, preprocess):
        query_image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(query_image).detach().cpu().flatten()
        return image_features


# Load functions for the Open CLIP model
def load_open_clip():
    """Get OpenAI CLIP model pool - kept for backward compatibility."""
    pool = get_model_pool('clip-openai')
    return pool


# Embed text using Open CLIP
def embed_text_open_clip(text):
    pool = get_model_pool('clip-openai')
    
    with pool.get_model() as (model, processor):
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )
        outputs = model(**inputs)
        return outputs.text_embeds


# Embed image using Open CLIP
def embed_image_open_clip(image):
    pool = get_model_pool('clip-openai')
    
    with pool.get_model() as (model, processor):
        inputs = processor(
            images=image,
            return_tensors="pt",
            padding=True,
        )
        outputs = model(**inputs)
        return outputs.image_embeds


def load_vit_webli():
    """Get WebLI model pool - kept for backward compatibility."""
    pool = get_model_pool('clip-vit-webli')
    return pool


# Embed text using webli
def embed_text_webli(text):
    pool = get_model_pool('clip-vit-webli')
    
    with pool.get_model() as (model, tokenizer, _):
        query_tokens = tokenizer(text)
        text_features = model.encode_text(query_tokens).detach().cpu().flatten()
        return text_features


# Embed image using webli
def embed_image_webli(image):
    pool = get_model_pool('clip-vit-webli')
    
    with pool.get_model() as (model, _, preprocess):
        query_image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(query_image).detach().cpu().flatten()
        return image_features


def load_vit_so400m():
    """Get SO400M model pool - kept for backward compatibility."""
    pool = get_model_pool('clip-vit-so400m')
    return pool


# Embed text using SO400M
def embed_text_so400m(text):
    pool = get_model_pool('clip-vit-so400m')
    
    with pool.get_model() as (model, tokenizer, _):
        query_tokens = tokenizer(text)
        text_features = model.encode_text(query_tokens)
        text_features = text_features.detach().cpu().flatten()
        return text_features


# Embed image using SO400M
def embed_image_so400m(image):
    pool = get_model_pool('clip-vit-so400m')
    
    with pool.get_model() as (model, _, preprocess):
        query_image = preprocess(image).unsqueeze(0)
        image_features = model.encode_image(query_image).detach().cpu().flatten()
        return image_features


def embed_text(text, model):
    """Embed text using the specified model with thread-safe pool access."""
    # Check if model is available
    if model not in available_models:
        l.logger.warning(f"Model {model} not available, using default {available_models[0]}")
        model = available_models[0]

    # Call the appropriate embedding function
    if 'clip-laion' in model:
        return embed_text_laion(text)
    elif 'clip-vit-webli' in model:
        return embed_text_webli(text)
    elif 'clip-openai' in model:
        return embed_text_open_clip(text)
    elif 'clip-vit-so400m' in model:
        return embed_text_so400m(text)
    else:
        l.logger.error(f"Unknown model: {model}")
        raise ValueError(f"Unknown model: {model}")


def embed_image(image, model):
    """Embed image using the specified model with thread-safe pool access."""
    # Check if model is available
    if model not in available_models:
        l.logger.warning(f"Model {model} not available, using default {available_models[0]}")
        model = available_models[0]

    # Call the appropriate embedding function
    if 'clip-laion' in model:
        return embed_image_laion(image)
    elif 'clip-vit-webli' in model:
        return embed_image_webli(image)
    elif 'clip-openai' in model:
        return embed_image_open_clip(image)
    elif 'clip-vit-so400m' in model:
        return embed_image_so400m(image)
    else:
        l.logger.error(f"Unknown model: {model}")
        raise ValueError(f"Unknown model: {model}")


def get_pool_status():
    """Get status of all model pools."""
    status = {}
    with pool_lock:
        for model_name, pool in model_pools.items():
            status[model_name] = {
                'initialized': pool.initialized,
                'available_instances': pool.pool.qsize() if pool.initialized else 0,
                'pool_size': pool.pool_size
            }
    return status


def warm_up_models(models_to_warm=None):
    """Warm up specified models by initializing their pools."""
    if models_to_warm is None:
        models_to_warm = available_models
    
    l.logger.info(f"Warming up models: {models_to_warm}")
    
    for model_name in models_to_warm:
        if model_name in available_models:
            l.logger.info(f"Warming up model pool: {model_name}")
            pool = get_model_pool(model_name)
            pool.initialize_pool()
        else:
            l.logger.warning(f"Skipping unknown model: {model_name}")


# Initialize the default model pool on startup
l.logger.info('Starting model initialization')
start_time = time.time()

# Check if warm-up is enabled and which models to warm up
if getattr(c, 'MODEL_WARM_UP_ON_STARTUP', True):
    models_to_warm = getattr(c, 'DEFAULT_MODELS_TO_WARM', ['clip-vit-so400m'])
    l.logger.info(f'Warming up models on startup: {models_to_warm}')
    warm_up_models(models_to_warm)
else:
    l.logger.info('Model warm-up on startup is disabled')

execution_time = time.time() - start_time
l.logger.info(f'Model initialization took: {execution_time:.6f} secs')
l.logger.info(f'Pool status: {get_pool_status()}')
