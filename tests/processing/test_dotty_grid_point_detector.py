# -*- coding: utf-8 -*-

"""
Tests for dotty grid implementation of PointDetector.
"""

import os
import cv2 as cv2
import numpy as np
import pytest
from sksurgeryimage.processing.dotty_grid_point_detector import DottyGridPointDetector


def test_dotty_grid_detector(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    image = cv2.imread('tests/data/calib-ucl-circles/circles-25x18-r40-s3.png')
    detector = DottyGridPointDetector(model_points, [133, 141, 308, 316])
    ids, object_points, image_points = detector.get_points(image)
    assert(model_points.shape[0] == ids.shape[0])
    assert(model_points.shape[0] == object_points.shape[0])
    assert(model_points.shape[0] == image_points.shape[0])


def __check_real_image(model_points, file_name):
    image = cv2.imread(file_name)
    detector = DottyGridPointDetector(model_points, [133, 141, 308, 316])
    ids, object_points, image_points = detector.get_points(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for counter in range(ids.shape[0]):
        cv2.putText(image, str(ids[counter][0]), (int(image_points[counter][0]), int(image_points[counter][1])), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    split_path = os.path.splitext(file_name)
    previous_dir = os.path.dirname(split_path[0])
    previous_dir = os.path.basename(previous_dir)
    base_name = os.path.basename(split_path[0])
    output_file = os.path.join('tests/output', base_name + '_' + previous_dir + '_labelled.png')
    cv2.imwrite(output_file, image)
    return ids.shape[0]


def test_dotty_grid_1(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_08/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(288 == number_of_points)


def test_dotty_grid_2(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_08/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(292 == number_of_points)


def test_dotty_grid_3(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_16/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(309 == number_of_points)


def test_dotty_grid_4(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_16/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(302 == number_of_points)


def test_dotty_grid_5(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_23/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(304 == number_of_points)


def test_dotty_grid_6(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_23/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(319 == number_of_points)


def test_dotty_grid_7(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_30/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(331 == number_of_points)


def test_dotty_grid_8(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_30/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(337 == number_of_points)


def test_dotty_grid_9(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_34/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(286 == number_of_points)


def test_dotty_grid_10(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_34/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(297 == number_of_points)


def test_dotty_grid_11(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_38/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(305 == number_of_points)


def test_dotty_grid_12(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_38/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(322 == number_of_points)


def test_dotty_grid_13(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_45/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(299 == number_of_points)


def test_dotty_grid_14(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_45/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(297 == number_of_points)


def test_dotty_grid_15(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_52/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(286 == number_of_points)


def test_dotty_grid_16(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_52/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(192 == number_of_points)


def test_dotty_grid_17(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_58/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(277 == number_of_points)


def test_dotty_grid_18(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_56_58/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(282 == number_of_points)


def test_dotty_grid_19(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_57_02/left_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(313 == number_of_points)


def test_dotty_grid_20(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    file_name = 'tests/data/calib-ucl-circles/snapshots/08_57_02/right_image.png'
    number_of_points = __check_real_image(model_points, file_name)
    assert(300 == number_of_points)
