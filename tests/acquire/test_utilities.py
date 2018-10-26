import pytest
import cv2
import numpy as np
from sksurgeryimage.acquire import utilities


def test_count_cameras():
    # Difficult to write a unit test as can't know how many cameras to expect.
    # For now, make sure no exceptions are thrown
    utilities.count_cameras()
    
def test_is_string_or_number():

    string = "test"
    integer = 1
    fp = 1.01
    list_invalid = [1,2,3]

    assert utilities.is_string_or_number(string)
    assert utilities.is_string_or_number(integer)
    assert utilities.is_string_or_number(fp)

    assert not utilities.is_string_or_number(list_invalid)


def test_validate_text_input():
    
    invalid = [1,2,3]

    with pytest.raises(TypeError):
        utilities.validate_text_input(invalid)


def test_prepare_text_overlay():

    frame_dims = (100,100,3)
    frame = np.empty(frame_dims, dtype=np.uint8)

    text_to_overlay = 1
    text_overlay_properties = utilities.prepare_cv2_text_overlay(text_to_overlay, frame)
    expected = ("1", (0, frame_dims[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255) )

    assert text_overlay_properties == expected

    scale = 10
    large_text_overlay_properties = utilities.prepare_cv2_text_overlay(text_to_overlay, frame, scale)
    expected_large = ("1", (0, frame_dims[0] - 10), cv2.FONT_HERSHEY_COMPLEX, scale, (255,255,255) )

    assert expected_large == large_text_overlay_properties
