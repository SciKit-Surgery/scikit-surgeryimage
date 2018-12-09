# coding=utf-8

"""
Tests for utilities.py
"""
import pytest
import numpy as np
import cv2 
from sksurgeryimage.utilities import utilities


def test_prepare_text_overlay():

    frame_dims = (100, 100, 3)
    frame = np.empty(frame_dims, dtype=np.uint8)

    text_to_overlay = 1
    text_overlay_properties = utilities.prepare_cv2_text_overlay(
        text_to_overlay, frame)
    expected = ("1", (0, frame_dims[0] - 10),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    assert text_overlay_properties == expected

    scale = 10
    large_text_overlay_properties = utilities.prepare_cv2_text_overlay(
        text_to_overlay, frame, scale)
    expected_large = (
        "1", (0, frame_dims[0] - 10), cv2.FONT_HERSHEY_COMPLEX, scale, (255, 255, 255))

    assert expected_large == large_text_overlay_properties

