# coding=utf-8

"""
Tests for PointDetector.
"""

import numpy as np
import pytest
from sksurgeryimage.calibration.point_detector import PointDetector


def test_cant_use_base_class():
    detector = PointDetector()
    image = np.ones((1, 1, 3), dtype=np.uint8)
    with pytest.raises(NotImplementedError):
        detector.get_points(image)


def test_invalid_because_image_is_none():
    detector = PointDetector()
    image = None
    with pytest.raises(TypeError):
        detector.get_points(image)


def test_invalid_because_image_is_not_ndarray():
    detector = PointDetector()
    image = "hello world"
    with pytest.raises(TypeError):
        detector.get_points(image)


def test_invalid_as_only_setting_intrinsics():
    with pytest.raises(ValueError):
        detector = PointDetector(camera_intrinsics=np.eye(3))


def test_invalid_as_only_setting_distortion_coefficients():
    with pytest.raises(ValueError):
        detector = PointDetector(distortion_coefficients=np.zeros((1, 5)))


def test_invalid_as_wrong_size_camera_matrix():
    with pytest.raises(ValueError):
        detector = PointDetector(camera_intrinsics=np.eye(4),
                                 distortion_coefficients=np.zeros((1, 5)))


def test_invalid_as_wrong_size_distortion_coefficients():
    with pytest.raises(ValueError):
        detector = PointDetector(camera_intrinsics=np.eye(3),
                                 distortion_coefficients=np.zeros((2, 5)))


def test_invalid_as_setting_none_camera_parameters():
    detector = PointDetector()
    with pytest.raises(ValueError):
        detector.set_camera_parameters(None, None)


def test_invalid_as_setting_invalid_camera_matrix():
    detector = PointDetector()
    with pytest.raises(ValueError):
        detector.set_camera_parameters(camera_intrinsics=np.eye(4))


def test_invalid_as_setting_invalid_camera_matrix():
    detector = PointDetector()
    with pytest.raises(ValueError):
        detector.set_camera_parameters(camera_intrinsics=np.eye(3),
                                       distortion_coefficients=np.zeros((2, 5)))
