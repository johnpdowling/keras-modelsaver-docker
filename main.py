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

import numpy as np

import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import tensorflow.contrib.tensorrt as trt

from fastapi import FastAPI, File, HTTPException
from starlette.requests import Request

model_folder = "/models"

os.makedirs(model_folder, exist_ok=True)
app = FastAPI()

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))

def save_model(model_name)
    model = ResNet50(weights=model_name)
    model.save(model_folder + "/" + model_name + ".h5"

def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen
    
@app.get("/")
def index():
    return "Hello World!"


@app.post("/model")
def model(request: Request, img_file: bytes=File(...)):
    data = {"success": False}

    if request.method == "POST":
        model = ResNet50(weights="imagenet")
        data["success"] = True
    # Return the data dictionary as a JSON response
    return data
