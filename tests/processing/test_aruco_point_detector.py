# coding=utf-8

"""
Tests for Aruco implementation of PointDetector.
"""

import cv2 as cv2
from cv2 import aruco
import numpy as np
import six
import pytest
from sksurgeryimage.processing.aruco_point_detector import ArucoPointDetector


def test_aruco_detector_without_model():
    image = cv2.imread('tests/data/processing/test-aruco.png')
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters_create()
    detector = ArucoPointDetector(dictionary, parameters, None, (1, 1))
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 12
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 0  # if we don't provide model, we can't output 3D coordinates.
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 12
    assert image_points.shape[1] == 2


def test_aruco_detector_with_model():
    image = cv2.imread('tests/data/processing/test-aruco.png')
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters_create()

    # The model should contain the ids and each 3D location for each model point.
    # Here we just generate dummy data.
    model = {}
    for i in range(0, 2000):
        model[i] = np.ones((1, 3))
        model[i][0][0] = i * 2
        model[i][0][1] = i * 3
        model[i][0][2] = 0

    detector = ArucoPointDetector(dictionary, parameters, model, (1, 1))
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 12
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 12
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 12
    assert image_points.shape[1] == 2
    six.print_('ArUco ids=' + str(ids))
    six.print_('ArUco object_points=' + str(object_points))
    six.print_('ArUco image_points=' + str(image_points))


def test_aruco_detector_with_point_not_in_model():
    image = cv2.imread('tests/data/processing/test-aruco.png')
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters_create()

    # The model should contain the ids and each 3D location for each model point.
    # Here we just generate dummy data.
    model = {}
    for i in range(0, 2):
        model[i] = np.ones((1, 3))
        model[i][0][0] = i * 2
        model[i][0][1] = i * 3
        model[i][0][2] = 0

    detector = ArucoPointDetector(dictionary, parameters, model, (1, 1))

    with pytest.raises(KeyError):
        _, _, _ = detector.get_points(image)
