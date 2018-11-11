# coding=utf-8

"""Functions to support deinterlacing, reinterlacing and
vertical destacking of 2D video frames."""

import numpy as np


def validate_interlaced_image_sizes(even_rows, odd_rows, interlaced):
    """
    Validates the sizes of the even_rows, odd_rows and interlaced images.

    1. Inputs must all be numpy images.
    2. Inputs must all have the same number of columns.
    3. Inputs must all have an even number of rows.
    4. even_rows and odd_rows must have the same number of rows.
    5. even_rows and odd_rows must have half the number of rows as interlaced.
    """

    if not isinstance(even_rows, np.ndarray):
        raise TypeError("even_rows is not a numpy array")

    if not isinstance(odd_rows, np.ndarray):
        raise TypeError("odd_rows is not a numpy array")

    if not isinstance(interlaced, np.ndarray):
        raise TypeError("interlaced is not a numpy array")

    if even_rows.shape[1] != odd_rows.shape[1]:
        raise ValueError("even_rows should have the same number "
                         + "of columns odd_rows")

    if odd_rows.shape[1] != interlaced.shape[1]:
        raise ValueError("odd_rows should have the same number of "
                         + "columns as interlaced")

    if even_rows.shape[0] % 2 != 0:
        raise ValueError("even_rows should have an even number of rows")

    if odd_rows.shape[0] % 2 != 0:
        raise ValueError("odd_rows should have an even number of rows")

    if interlaced.shape[0] % 2 != 0:
        raise ValueError("interlaced should have an even number of rows")

    if even_rows.shape[0] != odd_rows.shape[0]:
        raise ValueError("even_rows should have the same number of rows "
                         + "as odd_rows")

    if even_rows.shape[0] * 2 != interlaced.shape[0]:
        raise ValueError("even_rows output image should have half the number "
                         + "of rows as interlaced")


def interlace_to_preallocated(even_rows, odd_rows, interlaced):
    """
    Interlaces even_rows and odd_rows images into the interlaced image,
    where all inputs must be pre-allocated to the correct size.
    """
    validate_interlaced_image_sizes(even_rows, odd_rows, interlaced)

    interlaced[0::2] = even_rows
    interlaced[1::2] = odd_rows


def interlace_to_new(even_rows, odd_rows):
    """
    Interlaces even_rows and odd_rows images into a new output image.
    """
    if not isinstance(even_rows, np.ndarray):
        raise TypeError('even_rows is not a numpy array')

    if not isinstance(odd_rows, np.ndarray):
        raise TypeError('odd_rows is not a numpy array')

    new_height = even_rows.shape[0] + odd_rows.shape[0]
    new_dims = (new_height, even_rows.shape[1], even_rows.shape[2])
    interlaced = np.empty(new_dims, dtype=even_rows.dtype)

    # Contains further validation.
    interlace_to_preallocated(even_rows, odd_rows, interlaced)

    return interlaced


def deinterlace_to_preallocated(interlaced, even_rows, odd_rows):
    """
    Deinterlaces the interlaced image into even_rows and odd_rows images,
    which must be pre-allocated, and the correct size.
    """
    validate_interlaced_image_sizes(even_rows, odd_rows, interlaced)

    even_rows[:, :, :] = interlaced[0::2]
    odd_rows[:, :, :] = interlaced[1::2]


def deinterlace_to_view(interlaced):
    """
    Takes the interlaced image, and returns
    two new views of even_rows and odd_rows.

    :return: even_rows, odd_rows images
    """
    if not isinstance(interlaced, np.ndarray):
        raise TypeError('interlaced is not a numpy array')

    if interlaced.shape[0] % 2 != 0:
        raise ValueError("interlaced should have an even number of rows")

    even_rows = interlaced[0::2]
    odd_rows = interlaced[1::2]

    return even_rows, odd_rows


def deinterlace_to_new(interlaced):
    """
    Takes the interlaced image, and splits into two
    new images of even_rows and odd_rows.

    :return: even_rows, odd_rows images
    """
    if not isinstance(interlaced, np.ndarray):
        raise TypeError('interlaced is not a numpy array')

    if interlaced.shape[0] % 2 != 0:
        raise ValueError("interlaced should have an even number of rows")

    output_dims = (interlaced.shape[0]//2,
                   interlaced.shape[1],
                   interlaced.shape[2])

    even_rows = np.empty(output_dims, dtype=interlaced.dtype)
    odd_rows = np.empty(output_dims, dtype=interlaced.dtype)

    deinterlace_to_preallocated(interlaced, even_rows, odd_rows)

    return even_rows, odd_rows


def split_stacked_to_preallocated(stacked, top, bottom):
    """
    Splits a vertically stacked image, extracting the top and
    bottom halves, assuming images are the right size and pre-allocated.

    Useful if you have stereo 1080x1920 inputs, into an AJA Hi5-3D
    which stacks them vertically into 2 frames of 540x1920 in the
    same image.
    """
    validate_interlaced_image_sizes(top, bottom, stacked)

    top[:, :, :] = stacked[0:stacked.shape[0] // 2, :, :]
    bottom[:, :, :] = stacked[stacked.shape[0] // 2:, :, :]


def split_stacked_to_view(stacked):
    """
    Takes the input stacked image, and returns views that
    refer to the top and bottom half.

    :return: top_half, bottom_half images
    """
    if not isinstance(stacked, np.ndarray):
        raise TypeError('stacked is not a numpy array')

    if stacked.shape[0] % 2 != 0:
        raise ValueError("stacked should have an even number of rows")

    top_half = stacked[0:stacked.shape[0]//2, :, :]
    bottom_half = stacked[stacked.shape[0]//2:, :, :]

    return top_half, bottom_half


def split_stacked_to_new(stacked):
    """
    Takes the input stacked image, and extracts the top and bottom half.

    Useful if you have stereo 1080x1920 inputs, into an AJA Hi5-3D
    which stacks them vertically into 2 frames of 540x1920 in the
    same image.

    :return: top_half and bottom_half images
    """
    if not isinstance(stacked, np.ndarray):
        raise TypeError('stacked is not a numpy array')

    if stacked.shape[0] % 2 != 0:
        raise ValueError("stacked should have an even number of rows")

    output_dims = (stacked.shape[0]//2,
                   stacked.shape[1],
                   stacked.shape[2])

    top_half = np.empty(output_dims, dtype=stacked.dtype)
    bottom_half = np.empty(output_dims, dtype=stacked.dtype)

    split_stacked_to_preallocated(stacked,
                                  top_half,
                                  bottom_half)

    return top_half, bottom_half
