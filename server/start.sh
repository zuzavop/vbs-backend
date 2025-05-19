#!/bin/sh

uvicorn app:app --host 0.0.0.0 --port 8000 --proxy-headers --http h11 # --workers 2
#gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 2
