import numpy as np


def validate_interlaced_image_sizes(even_rows, odd_rows, interlaced):
    """
    Validates the relative sizes of the even_rows, odd_rows and interlaced images.

    1. Must all be the same width.
    2. Must all have an even number of rows.
    3. even_rows and odd_rows must have the same number of rows.
    4. even_rows and odd_rows must have half the number of rows as interlaced.

    Failures are indicated by throwing ValueError.

    :param even_rows: numpy image array, with even number of rows, for example 540 (rows) x 1920 (columns).
    :param odd_rows: numpy image array, with even number of rows, for example 540 (rows) x 1920 (columns).
    :param interlaced: numpy image array, with even number of rows, for example 1080 (rows) x 1920 (columns).
    :return: nothing
    """

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


def interlace(left, right):

    if not isinstance(left, np.ndarray):
        raise TypeError('Left input is not a numpy array')
        return False
    
    if not isinstance(right, np.ndarray):
        raise TypeError('Right input is not a numpy array')
        return False

    if left.shape != right.shape:
        raise ValueError('Left and Right arrays do not have the same dimensions')
        return False

    new_height = left.shape[0] + right.shape[0]
    new_dims = (new_height, left.shape[1], left.shape[2])

    interlaced = np.empty( new_dims, dtype = left.dtype)

    interlaced[0::2] = left
    interlaced[1::2] = right

    return interlaced

    
def deinterlace(interlaced):

    if not isinstance(interlaced, np.ndarray):
        raise TypeError('Input is not a numpy array')
        return False

    if interlaced.shape[0] % 2:
        raise ValueError("Interlaced array has an odd number of rows")
        return False
    

    # Using // for integer division
    output_dims = (interlaced.shape[0]//2, interlaced.shape[1], interlaced.shape[2])

    left = np.empty(output_dims, dtype = interlaced.dtype)
    right = np.empty(output_dims, dtype = interlaced.dtype)

    left = interlaced[0::2]
    right = interlaced[1::2]

    return left, right