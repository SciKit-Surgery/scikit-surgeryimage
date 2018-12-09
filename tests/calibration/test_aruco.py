# coding=utf-8

"""
Tests for aruco.py
"""

import cv2
from cv2 import aruco
import sksurgeryimage.calibration.aruco as ar


def test_extract_points():
    nx = 13
    ny = 10
    sx = 1300
    sy = 1000
    ss = 3
    ts = 2

    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    image, board = ar.make_charuco_board(dictionary, (nx, ny), (ss, ts), (sx, sy))

    number_of_markers, marker_corners, marker_ids, \
        chessboard_corners, chessboard_ids \
        = ar.detect_charuco_points(dictionary, image, board)

    expected_number = (nx - 1) * (ny - 1)
    assert number_of_markers == expected_number
    assert len(chessboard_corners) == expected_number
    assert len(chessboard_ids) == expected_number
    assert image.shape[0] == sy
    assert image.shape[1] == sx


def test_blank_charuco_board():

    dictionary = cv2.aruco.Dictionary_get(aruco.DICT_4X4_250)
    image, board = ar.make_charuco_board(dictionary, (13, 10), (3, 2), (1300, 1000))

    number_of_markers, marker_corners, marker_ids, \
        chessboard_corners, chessboard_ids \
        = ar.detect_charuco_points(dictionary, image, board)

    assert number_of_markers == 12*9

    edited = ar.erase_charuco_markers(image, marker_corners)
    corners = ar.draw_charuco_corners(image, chessboard_corners, chessboard_ids)

    cv2.imwrite('./tests/output/blanked_charuco_original.png', image)
    cv2.imwrite('./tests/output/blanked_charuco_edited.png', edited)
    cv2.imwrite('./tests/output/blanked_charuco_corners.png', corners)
