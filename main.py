import os
import cv2
import time
import numpy as np

import serial

import pyk4a
from pyk4a import Config, PyK4A

import torch
from model import SFNet
from torchvision import transforms as T

from gray2color import gray2color

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


