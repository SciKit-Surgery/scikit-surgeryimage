# coding=utf-8

"""
Tests for ChArUco + Chessboard implementation of PointDetector.
"""

import cv2 as cv2
import pytest
from sksurgeryimage.calibration.charuco_plus_chessboard_point_detector import CharucoPlusChessboardPointDetector
import sksurgeryimage.calibration.point_detector_utils as pdu
import sksurgeryimage.calibration.charuco as ch


def test_charuco_plus_chess_detector():

    calib_image = ch.make_charuco_with_chessboard()
    cv2.imwrite('tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_13x18.png', calib_image)
    return
    file_name = 'tests/data/calibration/pattern_4x4_26x19_5_4_with_inset_18x13.png'
    image = cv2.imread(file_name)
    detector = CharucoPlusChessboardPointDetector((19, 26), (5, 4), 25, (1, 1),
                                                  use_chessboard_inset=False,
                                                  number_of_chessboard_squares=(18, 13),
                                                  chessboard_square_size=3
                                                  )
    ids, object_points, image_points = detector.get_points(image)
    pdu.write_annotated_image(image, ids, image_points, file_name)
