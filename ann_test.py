__author__ = 'yujinke'
import numpy as np

import cv2

ann = cv2.ml.ANN_MLP_create()
ann.load("ann.xml")


