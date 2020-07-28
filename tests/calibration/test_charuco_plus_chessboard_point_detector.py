# coding=utf-8

"""
Tests for ChArUco + Chessboard implementation of PointDetector.
"""

import cv2 as cv2
import pytest
import numpy as np
from sksurgeryimage.calibration.charuco_plus_chessboard_point_detector import CharucoPlusChessboardPointDetector
import sksurgeryimage.calibration.point_detector_utils as pdu
import sksurgeryimage.calibration.charuco as ch


def test_charuco_plus_chess_detector(load_reference_charuco_chessboard_image):

    ref_img = load_reference_charuco_chessboard_image

    generated_image = ch.make_charuco_with_chessboard()
    output_file_name = 'tests/output/pattern_4x4_19x26_5_4_with_inset_9x14.png'
    cv2.imwrite(output_file_name, generated_image)
    generated_image = cv2.imread(output_file_name)

    input_file_name = 'tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_9x14.png'
    reference_image = cv2.imread(input_file_name)

    assert np.allclose(reference_image, generated_image)

    detector = CharucoPlusChessboardPointDetector(ref_img)
    ids_portrait, object_points_portrait, image_points_portrait = detector.get_points(reference_image)
    if ids_portrait.shape[0] > 0:
        pdu.write_annotated_image(reference_image, ids_portrait, image_points_portrait, 'tests/output/pattern_4x4_19x26_5_4_with_inset_9x14_portrait.png')

    input_file_name = 'tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_9x14_landscape.png'
    image = cv2.imread(input_file_name)
    ids_landscape, object_points_landscape, image_points_landscape = detector.get_points(image)
    if ids_landscape.shape[0] > 0:
        pdu.write_annotated_image(image, ids_landscape, image_points_landscape, 'tests/output/pattern_4x4_19x26_5_4_with_inset_9x14_landscape.png')

    assert ids_portrait.shape[0] == 468
    assert object_points_portrait.shape[0] == 468
    assert image_points_portrait.shape[0] == 468
    assert ids_landscape.shape[0] == 468
    assert object_points_landscape.shape[0] == 468
    assert image_points_landscape.shape[0] == 468

    assert np.allclose(ids_portrait, ids_landscape)
    assert np.allclose(object_points_portrait, object_points_landscape)


def test_charuco_plus_chess_invalid_because_no_reference_image():

    with pytest.raises(ValueError):
        detector = CharucoPlusChessboardPointDetector(None,
                                                      use_chessboard_inset=True,
                                                      number_of_chessboard_squares=None)


def test_charuco_plus_chess_invalid_because_reference_image_wrong_type():

    with pytest.raises(ValueError):
        detector = CharucoPlusChessboardPointDetector("wrong type",
                                                      use_chessboard_inset=True,
                                                      number_of_chessboard_squares=None)


def test_charuco_plus_chess_invalid_because_no_chessboard_squares(load_reference_charuco_chessboard_image):

    ref_img = load_reference_charuco_chessboard_image

    with pytest.raises(ValueError):
        detector = CharucoPlusChessboardPointDetector(ref_img,
                                                      use_chessboard_inset=True,
                                                      number_of_chessboard_squares=None)


def test_charuco_plus_chess_invalid_because_no_chessboard_size(load_reference_charuco_chessboard_image):

    ref_img = load_reference_charuco_chessboard_image

    with pytest.raises(ValueError):
        detector = CharucoPlusChessboardPointDetector(ref_img,
                                                      use_chessboard_inset=True,
                                                      chessboard_square_size=None)


def test_charuco_plus_chess_invalid_because_no_id_offset(load_reference_charuco_chessboard_image):

    ref_img = load_reference_charuco_chessboard_image

    with pytest.raises(ValueError):
        detector = CharucoPlusChessboardPointDetector(ref_img,
                                                      use_chessboard_inset=True,
                                                      chessboard_id_offset=None)


def test_charuco_plus_chess_invalid_because_id_offset_negative(load_reference_charuco_chessboard_image):

    ref_img = load_reference_charuco_chessboard_image

    with pytest.raises(ValueError):
        detector = CharucoPlusChessboardPointDetector(ref_img,
                                                      use_chessboard_inset=True,
                                                      chessboard_id_offset=-1)


def test_charuco_plus_chess_invalid_because_id_offset_too_small(load_reference_charuco_chessboard_image):

    ref_img = load_reference_charuco_chessboard_image

    # The default 26*19 grid will result in a maximum of 25*18 corners
    # from the ChArUco bit, so we must have an id_offset of at least 450.
    with pytest.raises(ValueError):
        detector = CharucoPlusChessboardPointDetector(ref_img,
                                                      use_chessboard_inset=True,
                                                      chessboard_id_offset=449)

    detector = CharucoPlusChessboardPointDetector(ref_img,
                                                  use_chessboard_inset=True,
                                                  chessboard_id_offset=450)


def test_charuco_plus_chess_invalid_because_no_chessboard_detected(load_reference_charuco_chessboard_image):

    ref_img = load_reference_charuco_chessboard_image

    roi = ref_img[0:400, 0:400, :]

    with pytest.raises(ValueError):
        # If user specifies in constructor that we are using a chessboard, then a chessboard must be detected.
        detector = CharucoPlusChessboardPointDetector(ref_img)
        _, _, _ = detector.get_points(roi)
