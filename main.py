import pandas as pd
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from multiprocessing import Process

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

torch.multiprocessing.freeze_support()

device = torch.device("cpu")

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using GPU:", torch.cuda.get_device_name())
# else:
#     device = torch.device("cpu")
#     print("CUDA is not available. Using CPU.")


## Defining the model

model = YOLO("yolov8m.yaml")  # build a new model from scratch

## training the model

results = model.train(data="config.yaml", epochs=300) 



  
