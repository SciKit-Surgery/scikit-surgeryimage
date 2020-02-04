# -*- coding: utf-8 -*-

"""
Tests for dotty grid implementation of PointDetector on metal images.
"""

import numpy as np
import pytest
import tests.processing.test_dotty_grid_utils as tdgu


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


def test_metal_7(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_10_45/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_8(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_10_45/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_9(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_11_22/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (223 == number_of_points)


def test_metal_10(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_11_22/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_11(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_11_56/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_12(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_11_56/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_13(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_12_04/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_14(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_12_04/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_15(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_12_24/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_16(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_12_24/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_17(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_12_44/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (223 == number_of_points)


def test_metal_18(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_12_44/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_19(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_13_18/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (222 == number_of_points)


def test_metal_20(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_13_18/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (222 == number_of_points)


def test_metal_21(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_13_46/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_22(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_13_46/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_23(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_14_26/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (222 == number_of_points)


def test_metal_24(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_14_26/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (223 == number_of_points)


def test_metal_25(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_14_58/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_26(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_14_58/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_27(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_16_59/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_28(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_16_59/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (224 == number_of_points)


def test_metal_29(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_17_36/left_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          True
                                          )
    assert (222 == number_of_points)


def test_metal_30(setup_dotty_metal_model):
    model_points = setup_dotty_metal_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-metal-1/14_17_36/right_image.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          True
                                          )
    assert (223 == number_of_points)
