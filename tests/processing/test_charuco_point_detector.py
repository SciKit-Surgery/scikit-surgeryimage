# coding=utf-8

"""
Tests for ChArUco implementation of PointDetector.
"""

import cv2 as cv2
from cv2 import aruco
import six
import pytest
from sksurgeryimage.processing.charuco_point_detector import CharucoPointDetector


def test_charuco_detector():
    image = cv2.imread('tests/data/processing/test-charuco.png')
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (13, 10), (3, 2))
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 108
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 108
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 108
    assert image_points.shape[1] == 2


def test_charuco_detector_with_masked_image():
    image = cv2.imread('tests/data/processing/test-charuco-blanked.png')
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (13, 10), (3, 2))
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 45
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 45
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 45
    assert image_points.shape[1] == 2


def test_charuco_detector_with_filtering():
    image = cv2.imread('tests/data/processing/pattern_4x4_19x26_5_4_with_inset_13x18_corrupted2.png')
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (19, 26), (5, 4), filtering=True)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 315
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 315
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 315
    assert image_points.shape[1] == 2


def test_charuco_detector_without_filtering():
    image = cv2.imread('tests/data/processing/pattern_4x4_19x26_5_4_with_inset_13x18_corrupted2.png')
    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (19, 26), (5, 4))
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 321
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 321
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 321
    assert image_points.shape[1] == 2
