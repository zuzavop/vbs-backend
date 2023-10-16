import logging
import sys

# logger config
formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

for name in logging.root.manager.loggerDict:
    for uvicorn_handler in logging.getLogger(name).handlers:
        uvicorn_handler.setFormatter(formatter)

logger = logging.getLogger('uvicorn')


# startup message
logger.info('Embedding Server Startup')
