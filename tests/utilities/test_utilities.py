# coding=utf-8

"""
Tests for utilities.py
"""
import pytest
import numpy as np
import cv2 
from sksurgeryimage.utilities import utilities


def test_is_string_or_number():

    string = "test"
    integer = 1
    fp = 1.01
    list_invalid = [1, 2, 3]

    assert utilities.is_string_or_number(string)
    assert utilities.is_string_or_number(integer)
    assert utilities.is_string_or_number(fp)

    assert not utilities.is_string_or_number(list_invalid)


def test_validate_text_input():

    invalid = [1, 2, 3]

    with pytest.raises(TypeError):
        utilities.validate_text_input(invalid)


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


def test_width_height_invalid_because_none_input():
    with pytest.raises(ValueError):
        utilities.validate_width_height(None)


def test_width_height_invalid_because_empty_input():
    with pytest.raises(ValueError):
        utilities.validate_width_height(())


def test_width_height_invalid_because_width_wrong_type():
    with pytest.raises(TypeError):
        utilities.validate_width_height(("", 2))


def test_width_height_invalid_because_width_too_low():
    with pytest.raises(ValueError):
        utilities.validate_width_height((0, 2))


def test_width_height_invalid_because_height_wrong_type():
    with pytest.raises(TypeError):
        utilities.validate_width_height((1, ""))


def test_width_height_invalid_because_height_too_low():
    with pytest.raises(ValueError):
        utilities.validate_width_height((1, 0))


def test_camera_matrix_invalid_because_wrong_type():
    with pytest.raises(TypeError):
        utilities.validate_camera_matrix(1)


def test_camera_matrix_invalid_because_not_two_dimensional():
    with pytest.raises(ValueError):
        utilities.validate_camera_matrix(np.ones((3, 3, 3)))


def test_camera_matrix_invalid_because_too_few_rows():
    with pytest.raises(ValueError):
        utilities.validate_camera_matrix(np.ones((1, 3)))


def test_camera_matrix_invalid_because_too_many_rows():
    with pytest.raises(ValueError):
        utilities.validate_camera_matrix(np.ones((4, 3)))


def test_camera_matrix_invalid_because_too_few_columns():
    with pytest.raises(ValueError):
        utilities.validate_camera_matrix(np.ones((3, 1)))


def test_camera_matrix_invalid_because_too_many_columns():
    with pytest.raises(ValueError):
        utilities.validate_camera_matrix(np.ones((3, 4)))


def test_distortion_coefficients_invalid_because_wrong_type():
    with pytest.raises(TypeError):
        utilities.validate_distortion_coefficients(1)


def test_distortion_coefficients_invalid_because_not_two_dimensional():
    with pytest.raises(ValueError):
        utilities.validate_distortion_coefficients(np.ones((3, 3, 3)))


def test_distortion_coefficients_invalid_because_too_many_rows():
    with pytest.raises(ValueError):
        utilities.validate_distortion_coefficients(np.ones((3, 4)))


def test_distortion_coefficients_invalid_because_too_few_columns():
    with pytest.raises(ValueError):
        utilities.validate_distortion_coefficients(np.ones((1, 3)))


def test_distortion_coefficients_invalid_because_number_of_columns_not_in_list():
    with pytest.raises(ValueError):
        # Should accept [4, 5, 8, 12, 14]
        utilities.validate_distortion_coefficients(np.ones((1, 6)))


def test_rotation_matrix_invalid_because_wrong_type():
    with pytest.raises(TypeError):
        utilities.validate_rotation_matrix(1)


def test_rotation_matrix_invalid_because_not_two_dimensional():
    with pytest.raises(ValueError):
        utilities.validate_rotation_matrix(np.ones((3, 3, 3)))


def test_rotation_matrix_invalid_because_too_many_rows():
    with pytest.raises(ValueError):
        utilities.validate_rotation_matrix(np.ones((4, 3)))


def test_rotation_matrix_invalid_because_too_many_columns():
    with pytest.raises(ValueError):
        utilities.validate_rotation_matrix(np.ones((3, 4)))


def test_rotation_matrix_invalid_because_too_few_rows():
    with pytest.raises(ValueError):
        utilities.validate_rotation_matrix(np.ones((2, 3)))


def test_rotation_matrix_invalid_because_too_few_columns():
    with pytest.raises(ValueError):
        utilities.validate_rotation_matrix(np.ones((3, 2)))


def test_translation_matrix_invalid_because_wrong_type():
    with pytest.raises(TypeError):
        utilities.validate_translation_column_vector(1)


def test_translation_matrix_invalid_because_not_two_dimensional():
    with pytest.raises(ValueError):
        utilities.validate_translation_column_vector(np.ones((3, 3, 3)))


def test_translation_matrix_invalid_because_too_many_rows():
    with pytest.raises(ValueError):
        utilities.validate_translation_column_vector(np.ones((4, 1)))


def test_translation_matrix_invalid_because_too_many_columns():
    with pytest.raises(ValueError):
        utilities.validate_translation_column_vector(np.ones((3, 4)))


def test_translation_matrix_invalid_because_too_few_rows():
    with pytest.raises(ValueError):
        utilities.validate_translation_column_vector(np.ones((2, 1)))



