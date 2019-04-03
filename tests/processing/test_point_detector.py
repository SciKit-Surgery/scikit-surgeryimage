# coding=utf-8

"""
Tests for PointDetector.
"""

import numpy as np
import pytest
from sksurgeryimage.processing.point_detector import PointDetector


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


def test_invalid_because_image_is_not_rgb_3_channel():
    detector = PointDetector()
    image = np.ones((1, 1, 1), dtype=np.uint8)
    with pytest.raises(ValueError):
        detector.get_points(image)
