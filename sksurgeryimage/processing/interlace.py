# coding=utf-8

"""Functions to support de-interlacing and re-interlacing of 2D video frames."""

import numpy as np


def validate_interlaced_image_sizes(even_rows, odd_rows, interlaced):
    """
    Validates the relative sizes of the even_rows, odd_rows and interlaced images.

    1. Inputs must all be numpy images, or throw TypeError.
    2. Must all be the same width, or throw ValueError.
    3. Must all have an even number of rows, or throw ValueError.
    4. even_rows and odd_rows must have the same number of rows, or throw ValueError.
    5. even_rows and odd_rows must have half the number of rows as interlaced, or throw ValueError.

    :param even_rows: numpy image array, with even number of rows, for example 540 (rows) x 1920 (columns).
    :param odd_rows: numpy image array, with even number of rows, for example 540 (rows) x 1920 (columns).
    :param interlaced: numpy image array, with even number of rows, for example 1080 (rows) x 1920 (columns).
    :return: nothing
    """

    if not isinstance(even_rows, np.ndarray):
        raise TypeError('even_rows is not a numpy array')

    if not isinstance(odd_rows, np.ndarray):
        raise TypeError('odd_rows is not a numpy array')

    if not isinstance(interlaced, np.ndarray):
        raise TypeError('interlaced is not a numpy array')

    if even_rows.shape[1] != odd_rows.shape[1]:
        raise ValueError("The even_rows image should have the same number of columns as the odd_rows image")

    if odd_rows.shape[1] != interlaced.shape[1]:
        raise ValueError("The odd_rows image should have the same number of columns as the interlaced image")

    if even_rows.shape[0] % 2 != 0:
        raise ValueError("The even_rows image should have an even number of rows.")

    if odd_rows.shape[0] % 2 != 0:
        raise ValueError("The odd_rows image should have an even number of rows.")

    if interlaced.shape[0] % 2 != 0:
        raise ValueError("The interlaced image should have an even number of rows.")

    if even_rows.shape[0] != odd_rows.shape[0]:
        raise ValueError("The even_rows image should have the same number of rows as the odd_rows image")

    if even_rows.shape[0] * 2 != interlaced.shape[0]:
        raise ValueError("The even_rows output image should have half the number of rows as the interlaced image.")

    if odd_rows.shape[0] * 2 != interlaced.shape[0]:
        raise ValueError("The odd_rows output image should have half the number of rows as the interlaced image.")


def interlace_preallocated_images(even_rows, odd_rows, interlaced):

    validate_interlaced_image_sizes(even_rows, odd_rows, interlaced)

    interlaced[0::2] = even_rows
    interlaced[1::2] = odd_rows


def interlace(even_rows, odd_rows):

    if not isinstance(even_rows, np.ndarray):
        raise TypeError('even_rows is not a numpy array')

    if not isinstance(odd_rows, np.ndarray):
        raise TypeError('odd_rows is not a numpy array')

    new_height = even_rows.shape[0] + odd_rows.shape[0]
    new_dims = (new_height, even_rows.shape[1], even_rows.shape[2])
    interlaced = np.empty(new_dims, dtype=even_rows.dtype)

    interlace_preallocated_images(even_rows, odd_rows, interlaced)

    return interlaced


def deinterlace_preallocated_images(interlaced, even_rows, odd_rows):

    validate_interlaced_image_sizes(even_rows, odd_rows, interlaced)

    even_rows[:, :, :] = interlaced[0::2]
    odd_rows[:, :, :] = interlaced[1::2]


def deinterlace(interlaced):

    if not isinstance(interlaced, np.ndarray):
        raise TypeError('interlaced is not a numpy array')

    output_dims = (interlaced.shape[0]//2, interlaced.shape[1], interlaced.shape[2])

    even_rows = np.empty(output_dims, dtype=interlaced.dtype)
    odd_rows = np.empty(output_dims, dtype=interlaced.dtype)

    deinterlace_preallocated_images(interlaced, even_rows, odd_rows)

    return even_rows, odd_rows
