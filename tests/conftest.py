# -*- coding: utf-8 -*-

import os
import pytest
import numpy as np
import cv2





@pytest.fixture(scope="function")
def setup_dotty_calibration_model():
    number_of_points = 18 * 25
    model_points = np.zeros((number_of_points, 6))
    counter = 0
    for y_index in range(18):
        for x_index in range(25):
            model_points[counter][0] = counter
            model_points[counter][1] = (x_index + 1) * 100
            model_points[counter][2] = (y_index + 1) * 100
            model_points[counter][3] = x_index * 5
            model_points[counter][4] = y_index * 5
            model_points[counter][5] = 0
            counter = counter + 1
    return model_points


@pytest.fixture(scope="function")
def setup_dotty_metal_model():
    number_of_points = 14 * 16
    model_points = np.zeros((number_of_points, 6))
    counter = 0
    for y_index in range(14):
        for x_index in range(16):
            model_points[counter][0] = counter
            model_points[counter][1] = (x_index + 1) * 80
            model_points[counter][2] = (y_index + 1) * 80
            model_points[counter][3] = x_index * 5
            model_points[counter][4] = y_index * 5
            model_points[counter][5] = 0
            counter = counter + 1
    return model_points
