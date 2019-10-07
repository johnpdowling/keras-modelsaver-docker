"""
Web server script that exposes REST endpoint and pushes images to Redis for classification by model server. Polls
Redis for response from model server.
Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""
import base64
import io
import json
import os
import time
import uuid

from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from PIL import Image
import redis

from fastapi import FastAPI, File, HTTPException
from starlette.requests import Request

app = FastAPI()

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))


@app.get("/")
def index():
    return "Hello World!"


@app.post("/model")
def model(request: Request, img_file: bytes=File(...)):
    data = {"success": False}

    if request.method == "POST":
        data["success"] = True
    # Return the data dictionary as a JSON response
    return data
