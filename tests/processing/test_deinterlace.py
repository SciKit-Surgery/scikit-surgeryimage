import pytest
import numpy as np
from sksurgeryimage.processing import interlace

@pytest.fixture
def create_valid_input():

    def _create_valid_input(rows, cols):

        dims = (rows, cols, 3)

        odd_row_value = 0
        even_row_value = 1

        interlaced = np.empty(dims, dtype=np.uint8)

        for idx in range(rows):
            if idx % 2:
                interlaced[idx, :, :] = odd_row_value
            else:
                interlaced[idx, :, :] = even_row_value

        return interlaced
    
    return _create_valid_input


def test_invalid_input():
    
    interlaced = None

    with pytest.raises(TypeError):
        interlace.deinterlace(interlaced)


def test_small_input(create_valid_input):

    rows = 20
    cols = 10

    interlaced = create_valid_input(rows, cols)
    
    left, right = interlace.deinterlace(interlaced)

    output_dims = (rows//2, cols, 3)

    expected_left = np.ones(output_dims, dtype=np.uint8)
    expected_right = np.zeros(output_dims, dtype=np.uint8)

    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_right)


def test_big_input(create_valid_input):

    rows = 540 * 2
    cols = 1920

    interlaced = create_valid_input(rows, cols)

    left, right = interlace.deinterlace(interlaced)

    output_dims = (rows//2, cols, 3)

    expected_left = np.ones(output_dims, dtype=np.uint8)
    expected_right = np.zeros(output_dims, dtype=np.uint8)

    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_right)

