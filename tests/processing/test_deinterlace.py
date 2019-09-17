import pytest
import numpy as np
import cv2 as cv2
from sksurgeryimage.processing import interlace

def test_empty_input_to_deinterlace_to_new():
    
    interlaced = None

    with pytest.raises(TypeError):
        interlace.deinterlace_to_new(interlaced)


def test_empty_input_to_deinterlace_to_view():

    interlaced = None

    with pytest.raises(TypeError):
        interlace.deinterlace_to_view(interlaced)


def test_odd_input_to_deinterlace_to_new(create_valid_interlaced_input):

    interlaced = create_valid_interlaced_input(5, 10)

    with pytest.raises(ValueError):
        interlace.deinterlace_to_new(interlaced)


def test_odd_input_to_deinterlace_to_view(create_valid_interlaced_input):

    interlaced = create_valid_interlaced_input(3, 10)

    with pytest.raises(ValueError):
        interlace.deinterlace_to_view(interlaced)


def test_small_input(create_valid_interlaced_input):

    rows = 20
    cols = 10

    interlaced = create_valid_interlaced_input(rows, cols)
    
    left, right = interlace.deinterlace_to_new(interlaced)

    output_dims = (rows//2, cols, 3)

    expected_left = np.ones(output_dims, dtype=np.uint8)
    expected_right = np.zeros(output_dims, dtype=np.uint8)

    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_right)


def test_big_input(create_valid_interlaced_input):

    rows = 540 * 2
    cols = 1920

    interlaced = create_valid_interlaced_input(rows, cols)

    left, right = interlace.deinterlace_to_new(interlaced)

    output_dims = (rows//2, cols, 3)

    expected_left = np.ones(output_dims, dtype=np.uint8)
    expected_right = np.zeros(output_dims, dtype=np.uint8)

    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_right)


def test_deinterlace_from_file():
    interlaced = cv2.imread('tests/data/processing/test-16x8-rgb.png')
    expected_even = cv2.imread('tests/data/processing/test-16x8-rgb-even.png')
    expected_odd = cv2.imread('tests/data/processing/test-16x8-rgb-odd.png')

    # Testing creating views
    even_view, odd_view = interlace.deinterlace_to_view(interlaced)
    np.testing.assert_array_equal(even_view, expected_even)
    np.testing.assert_array_equal(odd_view, expected_odd)

    # Testing creating new images
    even_new, odd_new = interlace.deinterlace_to_new(interlaced)
    np.testing.assert_array_equal(even_new, expected_even)
    np.testing.assert_array_equal(odd_new, expected_odd)
