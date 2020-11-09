# -*- coding: utf-8 -*-

"""
Tests for dotty grid implementation of PointDetector.
"""

import numpy as np
import cv2
import pytest
import tests.calibration.test_dotty_grid_utils as tdgu
import sksurgeryimage.calibration.dotty_grid_point_detector as dotty_pd

def test_tutorial_stuff():
    """ Code that is used in the dotty detector tutorial. """
    #Tutorial-section1-start
    number_of_dots = [18, 25]
    pixels_per_mm = 80
    dot_separation = 5

    model_points = dotty_pd.get_model_points(number_of_dots,
                                             pixels_per_mm,
                                             dot_separation)
    #Tutorial-section1-end

    assert model_points.shape == (450, 6)

    #Tutorial-section2-start
    # Location of the large dots in the pattern
    fiducial_indexes = [132, 142, 307, 317]

    # Image size
    reference_image_size = [1900, 2600]

    left_intrinsic_matrix = np.loadtxt("tests/data/calib-ucl-circles/calib.left.intrinsics.txt")
    left_distortion_matrix = np.loadtxt("tests/data/calib-ucl-circles/calib.left.distortion.txt")

    point_detector = \
        dotty_pd.DottyGridPointDetector(
            model_points,
            fiducial_indexes,
            left_intrinsic_matrix,
            left_distortion_matrix,
            reference_image_size=(reference_image_size[1],
                                    reference_image_size[0])
            )

    dot_pattern = cv2.imread("tests/data/calib-ucl-circles/circles-25x18-r50-s2.png")
    
    # Pass in the test image, in practice we would use a captured imaged instead.
    ids, object_points, image_points = point_detector.get_points(dot_pattern)
    #Tutorial-section2-end

    assert ids.shape == (430, 1)

    #Tutorial-section3-start
    for idx in range(ids.shape[0]):
        text = str(ids[idx][0])
        x = int(image_points[idx][0])
        y = int(image_points[idx][1])
        
        cv2.putText(dot_pattern,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

    #Tutorial-section3-end


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
    assert(376 == number_of_points)


def test_dotty_uncalibrated_4(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_54_23/right_image.png',
                                          'tests/data/calib-ucl-circles/calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.right.distortion.txt',
                                          )
    assert(369 == number_of_points)


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
    assert(356 == number_of_points)


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
    assert(363 == number_of_points)


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
    assert(394 == number_of_points)


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
    assert(366 == number_of_points)


def test_dotty_uncalibrated_17(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/snapshots-uncalibrated/08_55_13/left_image.png',
                                          'tests/data/calib-ucl-circles/calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/calib.left.distortion.txt',
                                          )
    assert(383 == number_of_points)


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
    assert(315 == number_of_points)


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
    assert(309 == number_of_points)


def test_calibration_4(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.4.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(354 == number_of_points)


def test_calibration_5(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.5.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(317 == number_of_points)


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
    assert(277 == number_of_points)


def test_calibration_8(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.left.images.8.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.left.distortion.txt',
                                          )
    assert(293 == number_of_points)


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
    assert (334 == number_of_points)


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
    assert (299 == number_of_points)


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
    assert (286 == number_of_points)


def test_calibration_18(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/10_54_44/calib.right.images.8.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (287 == number_of_points)


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
    assert (290 == number_of_points)


def test_calibration_21(setup_dotty_calibration_model):
    model_points = setup_dotty_calibration_model
    number_of_points = tdgu.__check_real_image(model_points,
                                          'tests/data/calib-ucl-circles/13_22_20/calib.right.images.9.png',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.intrinsics.txt',
                                          'tests/data/calib-ucl-circles/10_54_44/viking.calib.right.distortion.txt',
                                          )
    assert (304 == number_of_points)


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

