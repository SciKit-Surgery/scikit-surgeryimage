# coding=utf-8

"""
Various utilities, like preparing overlay text, validating image sizes,
camera matrix sizes, distortion coefficient sizes etc.
"""
import numpy as np
import os
import cv2


def prepare_cv2_text_overlay(overlay_text, frame, text_scale=1):
    """
    Return settings for text overlay on a cv2 frame.
    """
    validate_text_input(overlay_text)

    text = str(overlay_text)
    text_y_offset = 10
    text_location = (0, frame.shape[0] - text_y_offset)  # Bottom left
    text_colour = (255, 255, 255)

    text_overlay_properties = (
        text, text_location, cv2.FONT_HERSHEY_COMPLEX, text_scale, text_colour)

    return text_overlay_properties


def validate_text_input(overlay_text):
    """
    Raises an error if input isn't a string or number.

    :raises: TypeError
    """
    if not is_string_or_number(overlay_text):
        raise TypeError('Text overlay must be string or numeric')


def is_string_or_number(var):
    """
    Return True if the input variable is either a string or a numeric type.
    Return False otherwise.s
    """
    valid_types = (str, int, float)
    if isinstance(var, valid_types):
        return True

    return False


def validate_file_input(file_input):
    """
    Check if source file exists.
    """
    if os.path.isfile(file_input):
        return True

    raise ValueError('Input file:' + file_input + ' does not exist')


def validate_width_height(dims):
    """
    Checks if dims (width, height) is a valid specification,
    meaning width and height are both integers above zero.

    :param dims: (width, height)
    :raises: TypeError, ValueError
    """
    if dims is None:
        raise ValueError("Null input for (width, height).")

    width, height = dims

    if not isinstance(width, int):
        raise TypeError("Width should be an integer.")

    if not isinstance(height, int):
        raise TypeError("Height should be an integer.")

    if width < 1:
        raise ValueError("Width should be >= 1.")

    if height < 1:
        raise ValueError("Height should be >= 1.")


def validate_camera_matrix(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Camera matrix is not a numpy ndarray.")
    if len(matrix.shape) != 2:
        raise ValueError("Camera matrix should have 2 dimensions.")
    if matrix.shape[0] != 3:
        raise ValueError("Camera matrix should have 3 rows.")
    if matrix.shape[1] != 3:
        raise ValueError("Camera matrix should have 3 columns.")


def validate_distortion_coefficients(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Distortion coefficients are not a numpy ndarray.")
    if len(matrix.shape) != 2:
        raise ValueError("Camera matrix should have 2 dimensions.")
    if matrix.shape[0] != 1:
        raise ValueError("Distortion coefficients should have 1 row.")
    if matrix.shape[1] not in [4, 5, 8, 12, 14]:  # See OpenCV docs
        raise ValueError("Distortion coefficients should have 4, 5, 8, 12 or 14 columns.")


def validate_rotation_matrix(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Rotation matrix should be a numpy ndarray.")
    if len(matrix.shape) != 2:
        raise ValueError("Rotation matrix should have 2 dimensions.")
    if matrix.shape[0] != 3:
        raise ValueError("Rotation matrix should have 3 rows.")
    if matrix.shape[1] != 3:
        raise ValueError("Rotation matrix should have 3 columns.")


def validate_translation_matrix(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Translation matrix should be a numpy ndarray.")
    if len(matrix.shape) != 2:
        raise ValueError("Translation matrix should have 2 dimensions.")
    if matrix.shape[0] != 3:
        raise ValueError("Translation matrix  should have 3 rows.")
    if matrix.shape[1] != 1:
        raise ValueError("Translation matrix  should have 1 column.")
