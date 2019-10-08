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

def save_model(model_name):
    model = ResNet50(weights=model_name)
    model.save(model_folder + "/" + model_name + ".h5")

def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

def save_graph(model_name):
    # Clear any previous session.
    tf.keras.backend.clear_session()
    
    # This line must be executed before loading Keras model.
    tf.keras.backend.set_learning_phase(0) 
    model = load_model(model_folder + "/" + model_name + ".h5")
    session = tf.keras.backend.get_session()
    input_names = [t.op.name for t in model.inputs]
    output_names = [t.op.name for t in model.outputs]
    # freeze the model to a graph
    frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=model_folder)
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50
    )
    graph_io.write_graph(trt_graph, model_folder,
                         model_name + '.pb', as_text=False)
 
@app.get("/")
def index():
    return "Hello World!"


@app.post("/model")
def model(request: Request, img_file: bytes=File(...)):
    data = {"success": False}

    if request.method == "POST":
        model_name = "imagenet"
        save_model(model_name)
        save_graph(model_name)
        data["success"] = True
    # Return the data dictionary as a JSON response
    return data
