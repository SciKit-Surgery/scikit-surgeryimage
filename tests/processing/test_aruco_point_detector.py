# coding=utf-8

"""
Tests for Aruco implementation of PointDetector.
"""

import cv2 as cv2
from cv2 import aruco
import six
import pytest
from sksurgeryimage.processing.aruco_point_detector import ArucoPointDetector


def test_aruco_detector():
    image = cv2.imread('tests/data/processing/test-aruco.png')
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters_create()
    detector = ArucoPointDetector(dictionary, parameters, None, (1, 1))
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 12
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 12
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 12
    assert image_points.shape[1] == 2
