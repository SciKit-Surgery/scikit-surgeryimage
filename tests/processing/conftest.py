import pytest
import numpy as np

@pytest.fixture
def create_valid_interlaced_input():

    def _create_valid_interlaced_input(rows, cols):

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
    
    return _create_valid_interlaced_input


@pytest.fixture
def create_valid_deinterlace_inputs():

    def _create_valid_inputs(rows, cols):
        dims = (rows, cols, 3)

        left = np.ones(dims, dtype=np.uint8)
        right = np.zeros(dims, dtype=np.uint8)

        return left, right

    return _create_valid_inputs


@pytest.fixture
def create_input_and_output_arrays():

    def _create_input_and_output_arrays(rows, cols):
        dims = (rows, cols, 3)
        even_rows = np.ones(dims)
        odd_rows = np.ones(dims)
        combined_dims = (rows * 2, cols, 3)
        interlaced = np.ones(combined_dims)

        return even_rows, odd_rows, interlaced

    return _create_input_and_output_arrays
