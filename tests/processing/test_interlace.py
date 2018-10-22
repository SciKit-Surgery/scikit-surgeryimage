import numpy as np
import pytest
from sksurgeryimage.processing import interlace


@pytest.fixture
def create_valid_inputs():

    def _create_valid_inputs(rows, cols):
        dims = (rows, cols, 3)

        left = np.ones(dims, dtype=np.uint8)
        right = np.zeros(dims, dtype=np.uint8)

        return left, right

    return _create_valid_inputs


def test_empty_inputs():

    dims = (10, 10, 3)

    left = None
    right = np.ones(dims)

    with pytest.raises(TypeError):
        interlaced = interlace.interlace(left, right)


def test_inputs_arent_same_size():

    left_dims = (10, 10, 3)
    right_dims = (20, 20, 3)

    left = np.ones(left_dims)
    right = np.zeros(right_dims)

    with pytest.raises(ValueError):
        interlaced = interlace.interlace(left, right)


def test_small_inputs(create_valid_inputs):

    rows = 10
    cols = 10

    left, right = create_valid_inputs(rows, cols)
    interlaced = interlace.interlace(left, right)

    for idx in range(rows):
        left_idx = idx * 2
        right_idx = left_idx + 1

        np.testing.assert_array_equal(interlaced[left_idx, :, :], left[idx, :, :])
        np.testing.assert_array_equal(interlaced[right_idx, :, :], right[idx, :, :])
     
 
def test_hd_inputs(create_valid_inputs):

    rows = 1080
    cols = 1920

    left, right = create_valid_inputs(rows, cols)
    interlaced = interlace.interlace(left, right)

    for idx in range(rows):
        left_idx = idx * 2
        right_idx = left_idx + 1

        np.testing.assert_array_equal(interlaced[left_idx, :, :], left[idx, :, :])
        np.testing.assert_array_equal(interlaced[right_idx, :, :], right[idx, :, :])

