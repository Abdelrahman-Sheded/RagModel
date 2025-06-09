#!/bin/bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.test_api:app --bind=0.0.0.0:$PORT 