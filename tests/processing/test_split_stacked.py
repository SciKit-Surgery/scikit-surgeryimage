import numpy as np
import pytest
from sksurgeryimage.processing import interlace as i


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

    top, bottom = i.split_stacked(stacked)

    np.testing.assert_array_equal(top, expected_top)
    np.testing.assert_array_equal(bottom, expected_bottom)