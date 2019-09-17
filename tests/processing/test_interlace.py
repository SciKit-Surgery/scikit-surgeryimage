import numpy as np
import cv2 as cv2
import pytest
from sksurgeryimage.processing import interlace


def test_empty_left():

    dims = (10, 10, 3)

    left = None
    right = np.ones(dims)

    with pytest.raises(TypeError):
        interlace.interlace_to_new(left, right)


def test_empty_right():

    dims = (10, 10, 3)

    left = np.ones(dims)
    right = None

    with pytest.raises(TypeError):
        interlace.interlace_to_new(left, right)


def test_small_inputs(create_valid_deinterlace_inputs):

    rows = 10
    cols = 10

    left, right = create_valid_deinterlace_inputs(rows, cols)
    interlaced = interlace.interlace_to_new(left, right)

    for idx in range(rows):
        left_idx = idx * 2
        right_idx = left_idx + 1

        np.testing.assert_array_equal(interlaced[left_idx, :, :], left[idx, :, :])
        np.testing.assert_array_equal(interlaced[right_idx, :, :], right[idx, :, :])
     
 
def test_hd_inputs(create_valid_deinterlace_inputs):

    rows = 1080
    cols = 1920

    left, right = create_valid_deinterlace_inputs(rows, cols)
    interlaced = interlace.interlace_to_new(left, right)

    for idx in range(rows):
        left_idx = idx * 2
        right_idx = left_idx + 1

        np.testing.assert_array_equal(interlaced[left_idx, :, :], left[idx, :, :])
        np.testing.assert_array_equal(interlaced[right_idx, :, :], right[idx, :, :])


def test_interlace_from_file():
    even = cv2.imread('tests/data/processing/test-16x8-rgb-even.png')
    odd = cv2.imread('tests/data/processing/test-16x8-rgb-odd.png')
    expected_interlaced = cv2.imread('tests/data/processing/test-16x8-rgb.png')
    interlaced = interlace.interlace_to_new(even, odd)
    np.testing.assert_array_equal(interlaced, expected_interlaced)
