# coding=utf-8

"""
Tests for ChArUco implementation of PointDetector.
"""

import cv2
import pytest
import sksurgeryimage.calibration.point_detector_utils as pdu
from sksurgeryimage.calibration.charuco_point_detector import CharucoPointDetector


def test_charuco_detector():
    image = cv2.imread('tests/data/calibration/test-charuco.png')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (13, 10), (3, 2), legacy_pattern=True)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 108
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 108
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 108
    assert image_points.shape[1] == 2

    model = detector.get_model_points()
    assert model.shape[0] == 108


def test_charuco_detector_with_masked_image():
    image = cv2.imread('tests/data/calibration/test-charuco-blanked.png')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (13, 10), (3, 2), legacy_pattern=True)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 45
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 45
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 45
    assert image_points.shape[1] == 2


def test_charuco_detector_1():
    image_name = 'tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_13x18.png'
    image = cv2.imread(image_name)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (19, 26), (5, 4), legacy_pattern=True)
    ids, object_points, image_points = detector.get_points(image)
    pdu.write_annotated_image(image, ids, image_points, image_name)
    expected_number = 322
    assert ids.shape[0] == expected_number
    assert ids.shape[1] == 1
    assert object_points.shape[0] == expected_number
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == expected_number
    assert image_points.shape[1] == 2


def test_charuco_detector_2():
    image_name = "tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_9x14_landscape.png"
    image = cv2.imread(image_name)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (19, 26), (5, 4), legacy_pattern=True)
    ids, object_points, image_points = detector.get_points(image)
    pdu.write_annotated_image(image, ids, image_points, image_name)
    expected_number = 364
    assert ids.shape[0] == expected_number
    assert ids.shape[1] == 1
    assert object_points.shape[0] == expected_number
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == expected_number
    assert image_points.shape[1] == 2
