# coding=utf-8

"""
Tests for charuco.py
"""

import numpy as np
import cv2
import sksurgeryimage.calibration.charuco as charuco


def test_extract_points():
    nx = 13
    ny = 10
    sx = 1300
    sy = 1000
    ss = 3
    ts = 2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
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

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
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

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
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
