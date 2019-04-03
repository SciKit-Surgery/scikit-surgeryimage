# coding=utf-8

"""
Tests for chessboard implementation of PointDetector.
"""

import cv2 as cv2
import numpy as np
import pytest
from sksurgeryimage.processing.chessboard_point_detector import ChessboardPointDetector


def test_chessboard_detector():
    image = cv2.imread('tests/data/calib-ucl-chessboard/leftImage.png')
    detector = ChessboardPointDetector((13, 10), 3)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 130
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 130
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 130
    assert image_points.shape[1] == 2

    detector2 = ChessboardPointDetector((13, 10), 3, scale=(1, 2))
    ids2, object_points2, image_points2 = detector2.get_points(image)
    np.testing.assert_array_equal(ids, ids2)
    np.testing.assert_array_equal(object_points, object_points2)
    np.testing.assert_allclose(image_points, image_points2, 0.01, 2)


def test_non_chessboard_image():
    image = cv2.imread('tests/data/processing/j_eroded.png')
    detector = ChessboardPointDetector((13, 10), 3)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 0
    assert object_points.shape[0] == 0
    assert image_points.shape[0] == 0


def test_wrong_chessboard_image():
    image = cv2.imread('tests/data/calib-opencv/left01.jpg')
    detector = ChessboardPointDetector((13, 10), 3)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 0
    assert object_points.shape[0] == 0
    assert image_points.shape[0] == 0
