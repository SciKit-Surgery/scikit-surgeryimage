import numpy as np
import pytest
from sksurgeryimage.processing import interlace as i


@pytest.fixture
def create_small_images():

    def _create_small_images(rows, cols):
        dims = (rows, cols, 3)
        even_rows = np.ones(dims)
        odd_rows = np.ones(dims)
        combined_dims = (rows * 2, cols, 3)
        interlaced = np.ones(combined_dims)

        return even_rows, odd_rows, interlaced

    return _create_small_images


def test_first_arg_not_numpy(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    even_rows = None
    with pytest.raises(TypeError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)


def test_second_arg_not_numpy(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    odd_rows = None
    with pytest.raises(TypeError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)


def test_third_arg_not_numpy(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    interlaced = None
    with pytest.raises(TypeError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)


def test_mismatched_columns_even(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    even_rows = np.ones((4, 5, 3))
    with pytest.raises(ValueError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)


def test_mismatched_columns_odd(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    interlaced = np.ones((4, 5, 3))
    with pytest.raises(ValueError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)


def test_odd_rows_for_even_image(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    even_rows = np.ones((3, 4, 3))
    with pytest.raises(ValueError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)


def test_odd_rows_for_odd_image(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    odd_rows = np.ones((3, 4, 3))
    with pytest.raises(ValueError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)


def test_odd_rows_for_interlaced_image(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    interlaced = np.ones((9, 4, 3))
    with pytest.raises(ValueError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)


def test_mismatched_rows(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    odd_rows = np.ones((6, 4, 3))
    with pytest.raises(ValueError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)


def test_even_image_not_half_of_interlaced(create_small_images):
    even_rows, odd_rows, interlaced = create_small_images(4, 4)
    interlaced = np.ones((6, 4, 3))
    with pytest.raises(ValueError):
        i.validate_interlaced_image_sizes(even_rows,
                                          odd_rows,
                                          interlaced)