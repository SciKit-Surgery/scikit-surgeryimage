# -*- coding: utf-8 -*-

"""
Tests for dotty grid implementation of PointDetector.
"""

import os
import datetime
import logging
import cv2 as cv2
import numpy as np
import pytest
from sksurgeryimage.processing.dotty_grid_point_detector import DottyGridPointDetector


def __check_real_image(model_points,
                       image_file_name,
                       intrinsics_file_name,
                       distortion_file_name
                       ):
    logging.basicConfig(level=logging.DEBUG)
    image = cv2.imread(image_file_name)
    intrinsics = np.loadtxt(intrinsics_file_name)
    distortion = np.loadtxt(distortion_file_name)

    detector = DottyGridPointDetector(model_points,
                                      [133, 141, 308, 316],
                                      intrinsics,
                                      distortion
                                      )

    time_before = datetime.datetime.now()

    ids, object_points, image_points = detector.get_points(image)

    time_after = datetime.datetime.now()
    time_diff = time_after - time_before

    print("__check_real_image:time_diff=" + str(time_diff))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for counter in range(ids.shape[0]):
        cv2.putText(image, str(ids[counter][0]), (int(image_points[counter][0]), int(image_points[counter][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    split_path = os.path.splitext(image_file_name)
    previous_dir = os.path.dirname(split_path[0])
    previous_dir = os.path.basename(previous_dir)
    base_name = os.path.basename(split_path[0])
    output_file = os.path.join('tests/output', base_name + '_' + previous_dir + '_labelled.png')
    cv2.imwrite(output_file, image)
    return ids.shape[0]


def test_dotty_uncalibrated_1(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_13/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(373 == number_of_points)


def test_dotty_uncalibrated_2(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_13/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(363 == number_of_points)


def test_dotty_uncalibrated_3(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_23/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(374 == number_of_points)


def test_dotty_uncalibrated_4(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_23/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(362 == number_of_points)


def test_dotty_uncalibrated_5(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_29/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(352 == number_of_points)


def test_dotty_uncalibrated_6(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_29/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(353 == number_of_points)


def test_dotty_uncalibrated_7(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_39/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(376 == number_of_points)


def test_dotty_uncalibrated_8(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_39/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(362 == number_of_points)


def test_dotty_uncalibrated_9(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_44/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(404 == number_of_points)


def test_dotty_uncalibrated_10(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_44/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(391 == number_of_points)


def test_dotty_uncalibrated_11(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_51/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(351 == number_of_points)


def test_dotty_uncalibrated_12(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_51/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(344 == number_of_points)


def test_dotty_uncalibrated_13(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_57/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(358 == number_of_points)


def test_dotty_uncalibrated_14(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_57/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(354 == number_of_points)


def test_dotty_uncalibrated_15(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_08/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(363 == number_of_points)


def test_dotty_uncalibrated_16(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_08/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(364 == number_of_points)


def test_dotty_uncalibrated_17(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_13/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(380 == number_of_points)


def test_dotty_uncalibrated_18(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_13/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(371 == number_of_points)


def test_dotty_uncalibrated_19(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_20/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(364 == number_of_points)


def test_dotty_uncalibrated_20(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = __check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_20/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(346 == number_of_points)

