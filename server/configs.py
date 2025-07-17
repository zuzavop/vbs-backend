# -*- coding: utf-8 -*-
#!/usr/bin/env python3

DATABASE_ROOT = '/data/vbs/'
DATABASE_IMAGES = '/data/vbs/images/'

BASE_MODEL = 'clip-vit-so400m'
BASE_DATASET = 'MVK'

BASE_K = 1000
BASE_ADD_FEATURES = 0
BASE_DOWNLOADING_SPEED_UP = 1
BASE_ROUNDING_PRECISION = 5
BASE_MULTIPLICATION = 1
BASE_MULTIPLIER = 10000
BASE_MAX_LABELS = 10
BASE_LIFE_LOG = False

# Model pool configuration
MODEL_POOL_SIZE = 4
MODEL_WARM_UP_ON_STARTUP = True
DEFAULT_MODELS_TO_WARM = ['clip-vit-so400m']  # Models to initialize on startup
