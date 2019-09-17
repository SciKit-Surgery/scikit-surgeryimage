# coding=utf-8

"""
Tests for charuco.py
"""

import numpy as np
import cv2
from cv2 import aruco
import sksurgeryimage.calibration.charuco as charuco


def test_extract_points():
    nx = 13
    ny = 10
    sx = 1300
    sy = 1000
    ss = 3
    ts = 2

    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    image, board = charuco.make_charuco_board(dictionary, (nx, ny), (ss, ts), (sx, sy))

    marker_corners, marker_ids, \
        chessboard_corners, chessboard_ids \
        = charuco.detect_charuco_points(dictionary, board, image)

    expected_number = (nx - 1) * (ny - 1)
    assert len(chessboard_corners) == expected_number
    assert len(chessboard_ids) == expected_number
    assert image.shape[0] == sy
    assert image.shape[1] == sx


def test_edit_out_charuco_board():

    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    image, board = charuco.make_charuco_board(dictionary, (13, 10), (3, 2), (1300, 1000))

    marker_corners, marker_ids, \
        chessboard_corners, chessboard_ids \
        = charuco.detect_charuco_points(dictionary, board, image)

    assert len(chessboard_corners) == 12*9

    edited = charuco.erase_charuco_markers(image, marker_corners)
    corners = charuco.draw_charuco_corners(image, chessboard_corners, chessboard_ids)

    cv2.imwrite('tests/output/blanked_charuco_original.png', image)
    cv2.imwrite('tests/output/blanked_charuco_edited.png', edited)
    cv2.imwrite('tests/output/blanked_charuco_corners.png', corners)


def test_dont_fail_if_no_markers_actually_present():

    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    image, board = charuco.make_charuco_board(dictionary,
                                         (13, 10),
                                         (3, 2),
                                         (1300, 1000))

    input_image_to_check = np.zeros((100, 50, 1), dtype=np.uint8)
    marker_corners, marker_ids, \
        chessboard_corners, chessboard_ids \
        = charuco.detect_charuco_points(dictionary, board, input_image_to_check)

    assert not marker_corners
    assert not marker_ids
    assert not chessboard_corners
    assert not chessboard_ids

    annotated_image = charuco.draw_charuco_corners(input_image_to_check,
                                              chessboard_corners,
                                              chessboard_ids)

    assert annotated_image.shape == input_image_to_check.shape


def test_detect_charuco_points_with_png_files():

    nx = 19
    ny = 26
    ss = 5
    ts = 4
    sx = 1300
    sy = 950

    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    _, board = charuco.make_charuco_board(dictionary, (nx, ny), (ss, ts), (sx, sy))

    # All markers in the image are at the correct positions.
    image = cv2.imread('tests/data/processing/pattern_4x4_19x26_5_4_with_inset_13x18.png')
    marker_corners, marker_ids, \
        chessboard_corners, chessboard_ids \
        = charuco.detect_charuco_points(dictionary, board, image, filtering=True)

    expected_number = 322
    assert len(chessboard_corners) == expected_number
    assert len(chessboard_ids) == expected_number
    assert image.shape[0] == sy
    assert image.shape[1] == sx

    # In this image Marker 228 is replaced by another marker so not detected,
    # so corners id=0 and id=18. No marker was filtered out.
    image = cv2.imread('tests/data/processing/pattern_4x4_19x26_5_4_with_inset_13x18_corrupted1.png')
    marker_corners, marker_ids, \
        chessboard_corners, chessboard_ids \
        = charuco.detect_charuco_points(dictionary, board, image, filtering=True)

    expected_number = 320
    assert len(chessboard_corners) == expected_number
    assert len(chessboard_ids) == expected_number
    assert image.shape[0] == sy
    assert image.shape[1] == sx

    # In this image Marker 9 at the bottom right was replaced by Marker 228
    # which was first detected to be at the wrong place so filtered out together with Marker 238.
    # The Marker 228 at the right place is actually not filtered out.
    image = cv2.imread('tests/data/processing/pattern_4x4_19x26_5_4_with_inset_13x18_corrupted2.png')
    marker_corners, marker_ids, \
        chessboard_corners, chessboard_ids \
        = charuco.detect_charuco_points(dictionary, board, image, filtering=True)

    expected_number = 315
    assert len(chessboard_corners) == expected_number
    assert len(chessboard_ids) == expected_number
    assert image.shape[0] == sy
    assert image.shape[1] == sx

    # Without filtering, Marker 9 is not found so only one id (the bottom right one)
    # less. But id=0 and id=18 will be at the wrong places.
    marker_corners, marker_ids, \
        chessboard_corners, chessboard_ids \
        = charuco.detect_charuco_points(dictionary, board, image)

    expected_number = 321
    assert len(chessboard_corners) == expected_number
    assert len(chessboard_ids) == expected_number

    corners_detected = charuco.draw_charuco_corners(image, chessboard_corners, chessboard_ids)
    cv2.imwrite('tests/output/corners_detected.png', corners_detected)
