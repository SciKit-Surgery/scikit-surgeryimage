import numpy as np
import cv2 as cv2
import pytest
from sksurgeryimage.processing import interlace as i


def test_empty_input_to_split_stacked_to_new():

    stacked = None

    with pytest.raises(TypeError):
        i.split_stacked_to_new(stacked)


def test_empty_input_to_split_stacked_to_view():

    stacked = None

    with pytest.raises(TypeError):
        i.split_stacked_to_view(stacked)


def test_odd_input_to_split_stacked_to_new():

    dims = [5, 10, 3]
    stacked = np.ndarray(dims, dtype=np.uint8)

    with pytest.raises(ValueError):
        i.split_stacked_to_new(stacked)


def test_odd_input_to_split_stacked_to_view():

    dims = [5, 10, 3]
    stacked = np.ndarray(dims, dtype=np.uint8)

    with pytest.raises(ValueError):
        i.split_stacked_to_view(stacked)


def test_small_image_split():
    rows = 20
    cols = 10
    dims = [rows, cols, 3]
    expected_top = np.ndarray(dims, dtype=np.uint8)
    expected_top.fill(2)
    expected_bottom = np.ndarray(dims, dtype=np.uint8)
    expected_bottom.fill(3)
    stacked = np.ndarray([rows * 2, cols, 3], dtype=np.uint8)
    stacked[0:20, :, :] = expected_top
    stacked[20:, :, :] = expected_bottom

    top, bottom = i.split_stacked_to_new(stacked)

    np.testing.assert_array_equal(top, expected_top)
    np.testing.assert_array_equal(bottom, expected_bottom)


def test_split_from_file():
    stacked = cv2.imread('tests/data/processing/test-16x8-rgb.png')
    expected_top = cv2.imread('tests/data/processing/test-16x8-rgb-top.png')
    expected_bottom = cv2.imread('tests/data/processing/test-16x8-rgb-bottom.png')

    # Testing creating views
    top_view, bottom_view = i.split_stacked_to_view(stacked)
    np.testing.assert_array_equal(top_view, expected_top)
    np.testing.assert_array_equal(bottom_view, expected_bottom)

    # Testing creating new images
    top_new, bottom_new = i.split_stacked_to_new(stacked)
    np.testing.assert_array_equal(top_new, expected_top)
    np.testing.assert_array_equal(bottom_new, expected_bottom)