# -*- coding: utf-8 -*-

"""
Tests for dotty grid implementation of PointDetector.
"""

import numpy as np
import pytest
import tests.processing.test_dotty_grid_utils as tdgu


def test_dotty_uncalibrated_1(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_13/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(375 == number_of_points)


def test_dotty_uncalibrated_2(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_13/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(368 == number_of_points)


def test_dotty_uncalibrated_3(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_23/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(375 == number_of_points)


def test_dotty_uncalibrated_4(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_23/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(368 == number_of_points)


def test_dotty_uncalibrated_5(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_29/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(357 == number_of_points)


def test_dotty_uncalibrated_6(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_29/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(355 == number_of_points)


def test_dotty_uncalibrated_7(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_39/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(371 == number_of_points)


def test_dotty_uncalibrated_8(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_39/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(362 == number_of_points)


def test_dotty_uncalibrated_9(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_44/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(409 == number_of_points)


def test_dotty_uncalibrated_10(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_44/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(395 == number_of_points)


def test_dotty_uncalibrated_11(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_51/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(352 == number_of_points)


def test_dotty_uncalibrated_12(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_51/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(346 == number_of_points)


def test_dotty_uncalibrated_13(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_57/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(363 == number_of_points)


def test_dotty_uncalibrated_14(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_57/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(359 == number_of_points)


def test_dotty_uncalibrated_15(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_08/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(366 == number_of_points)


def test_dotty_uncalibrated_16(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_08/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(367 == number_of_points)


def test_dotty_uncalibrated_17(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_13/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(384 == number_of_points)


def test_dotty_uncalibrated_18(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_13/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(375 == number_of_points)


def test_dotty_uncalibrated_19(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_20/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(368 == number_of_points)


def test_dotty_uncalibrated_20(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_20/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(353 == number_of_points)


def test_calibration_0(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.0.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(314 == number_of_points)


def test_calibration_1(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.1.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(335 == number_of_points)


def test_calibration_2(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.2.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(318 == number_of_points)


def test_calibration_3(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.3.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(311 == number_of_points)


def test_calibration_4(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.4.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(353 == number_of_points)


def test_calibration_5(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.5.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(316 == number_of_points)


def test_calibration_6(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.6.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(330 == number_of_points)


def test_calibration_7(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.7.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(280 == number_of_points)


def test_calibration_8(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.8.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(294 == number_of_points)


def test_calibration_9(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.9.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(299 == number_of_points)


def test_calibration_10(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.0.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert(312 == number_of_points)


def test_calibration_11(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.1.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (335 == number_of_points)


def test_calibration_12(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.2.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (317 == number_of_points)


def test_calibration_13(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.3.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (301 == number_of_points)


def test_calibration_14(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.4.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (355 == number_of_points)


def test_calibration_15(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.5.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (312 == number_of_points)


def test_calibration_16(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.6.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (336 == number_of_points)


def test_calibration_17(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.7.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (288 == number_of_points)


def test_calibration_18(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.8.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (288 == number_of_points)


def test_calibration_19(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.9.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (292 == number_of_points)


def test_calibration_20(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/13_22_20/calib.left.images.9.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert (293 == number_of_points)


def test_calibration_21(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/13_22_20/calib.right.images.9.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (305 == number_of_points)


def test_metal_1(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_08_32/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_2(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_08_32/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_3(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_09_35/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (223 == number_of_points)


def test_metal_4(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_09_35/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_5(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_10_22/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_6(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_10_22/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)

