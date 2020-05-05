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


def test_charuco_plus_chess_detector():

    generated_image = ch.make_charuco_with_chessboard()
    output_file_name = 'tests/output/pattern_4x4_19x26_5_4_with_inset_9x14.png'
    cv2.imwrite(output_file_name, generated_image)
    generated_image = cv2.imread(output_file_name)

    input_file_name = 'tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_9x14.png'
    reference_image = cv2.imread(input_file_name)

    assert np.allclose(reference_image, generated_image)

    detector = CharucoPlusChessboardPointDetector()
    ids_portrait, object_points_portrait, image_points_portrait = detector.get_points(reference_image)
    if ids_portrait.shape[0] > 0:
        pdu.write_annotated_image(reference_image, ids_portrait, image_points_portrait, input_file_name)

    input_file_name = 'tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_9x14_landscape.png'
    image = cv2.imread(input_file_name)
    ids_landscape, object_points_landscape, image_points_landscape = detector.get_points(image)
    if ids_landscape.shape[0] > 0:
        pdu.write_annotated_image(image, ids_landscape, image_points_landscape, input_file_name)

    assert ids_portrait.shape[0] == 468
    assert object_points_portrait.shape[0] == 468
    assert image_points_portrait.shape[0] == 468
    assert ids_landscape.shape[0] == 468
    assert object_points_landscape.shape[0] == 468
    assert image_points_landscape.shape[0] == 468

    assert np.allclose(ids_portrait, ids_landscape)
    assert np.allclose(object_points_portrait, object_points_landscape)
