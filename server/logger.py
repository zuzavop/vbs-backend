import logging
import sys

# logger config
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger('uvicorn')
logger.addHandler(handler)

# startup message
logger.info('Embedding Server Startup')
