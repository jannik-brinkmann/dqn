import cv2
import numpy as np

from datetime import datetime
import random
import torch
import os

CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'weights')


def create_dirs():
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

