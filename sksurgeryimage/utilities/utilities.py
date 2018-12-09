# coding=utf-8

"""
Various utilities, like preparing overlay text.
"""
import cv2
import sksurgerycore.utilities.validate as scv


def prepare_cv2_text_overlay(overlay_text, frame, text_scale=1):
    """
    Return settings for text overlay on a cv2 frame.
    """
    scv.validate_is_string_or_number(overlay_text)

    text = str(overlay_text)
    text_y_offset = 10
    text_location = (0, frame.shape[0] - text_y_offset)  # Bottom left
    text_colour = (255, 255, 255)

    text_overlay_properties = (
        text, text_location, cv2.FONT_HERSHEY_COMPLEX, text_scale, text_colour)

    return text_overlay_properties
