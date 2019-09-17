# coding=utf-8

"""
Test functuons for morphological_operators.py
"""

import numpy as np
import cv2 as cv2
import pytest
from sksurgeryimage.processing import morphological_operators as mo

# Image j.png from: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html


def test_erode_cross_3():
    """ Just a regression test really, and ensuring that the code is called at least once. """
    original = cv2.imread('tests/data/processing/j.png')
    expected = cv2.imread('tests/data/processing/j_eroded.png')
    output = mo.erode_with_cross(original, iterations=1)
    np.testing.assert_array_equal(output, expected)


def test_dilate_cross_3():
    """ Just a regression test really, and ensuring that the code is called at least once. """
    original = cv2.imread('tests/data/processing/j.png')
    expected = cv2.imread('tests/data/processing/j_dilated.png')
    output = mo.dilate_with_cross(original, iterations=10)
    np.testing.assert_array_equal(output, expected)
