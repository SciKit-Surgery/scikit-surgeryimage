# coding=utf-8

"""
Various utilities, like checking strings, numbers, preparing overlay text etc.
"""
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
