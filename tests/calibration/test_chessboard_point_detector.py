# coding=utf-8

"""
Tests for chessboard implementation of PointDetector.
"""
import os
import cv2 as cv2
import numpy as np
import sksurgeryimage.calibration.point_detector_utils as pdu
from sksurgeryimage.calibration.chessboard_point_detector import ChessboardPointDetector


def test_chessboard_detector():
    image_file_name = 'tests/data/calib-ucl-chessboard/leftImage.png'
    image = cv2.imread(image_file_name)
    detector = ChessboardPointDetector((13, 10), 3)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 130
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 130
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 130
    assert image_points.shape[1] == 2
    annotated_image = pdu.get_annotated_image(image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_chessboard_detector.png"), annotated_image)

    # For the above test, I'm expected points to be numbered left to right, top to bottom.
    assert image_points[1][0] > image_points[0][0]
    assert image_points[13][1] > image_points[0][1]

    # Here, we are testing that when we scale the image,
    # we get the same 2D points back (within tolerance).
    # This is because we sometimes use interlaced HD, which
    # is 1920x540, so we scale the image up, then detect points,
    # then scale the points back down, so the points are in the
    # original image coordinate system. i.e. same as above.
    detector2 = ChessboardPointDetector((13, 10), 3, scale=(1, 2))
    ids2, object_points2, image_points2 = detector2.get_points(image)
    np.testing.assert_array_equal(ids, ids2)
    np.testing.assert_array_equal(object_points, object_points2)
    np.testing.assert_allclose(image_points, image_points2, 0.01, 2)

    model = detector2.get_model_points()
    assert model.shape[0] == 130


def test_chessboard_detector_rotated_90():
    image_file_name = 'tests/data/calib-ucl-chessboard/leftImage.png'
    image = cv2.imread(image_file_name)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    detector = ChessboardPointDetector((13, 10), 3)
    ids, object_points, image_points = detector.get_points(rotated_image)
    assert ids.shape[0] == 130
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 130
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 130
    assert image_points.shape[1] == 2
    annotated_image = pdu.get_annotated_image(rotated_image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_chessboard_detector_rotated_90.png"), annotated_image)

    # For the above test, I'm expected points to be numbered top-to-bottom, right-to-left
    assert image_points[1][1] > image_points[0][1]
    assert image_points[13][0] < image_points[0][0]


def test_chessboard_detector_rotated_180():
    image_file_name = 'tests/data/calib-ucl-chessboard/leftImage.png'
    image = cv2.imread(image_file_name)
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    detector = ChessboardPointDetector((13, 10), 3)
    ids, object_points, image_points = detector.get_points(rotated_image)
    assert ids.shape[0] == 130
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 130
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 130
    assert image_points.shape[1] == 2
    annotated_image = pdu.get_annotated_image(rotated_image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_chessboard_detector_rotated_180.png"), annotated_image)

    # We know the original image (unrotated image) has origin at top-left.
    # So, origin should now be bottom-right
    assert image_points[0][0] > image_points[129][0]
    assert image_points[0][1] > image_points[129][1]

    # For the above test, I'm expected points to be numbered right-to-left, bottom-to-top
    # (i.e. upside down)
    assert image_points[1][0] < image_points[0][0]
    assert image_points[13][1] < image_points[0][1]


def test_chessboard_black_corners_10x8():
    """
    The above tests may have worked because 2 corners are black, and
    two corners are white. This test has all black corners.
    """
    image_file_name = 'tests/data/calibration/test-chessboard-10x8.png'
    image = cv2.imread(image_file_name)
    detector = ChessboardPointDetector((10, 8), 15)
    ids, object_points, image_points = detector.get_points(image)
    expected_number_of_points = 80
    assert ids.shape[0] == expected_number_of_points
    assert ids.shape[1] == 1
    assert object_points.shape[0] == expected_number_of_points
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == expected_number_of_points
    assert image_points.shape[1] == 2
    annotated_image = pdu.get_annotated_image(image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_chessboard_black_corners_10x8_original.png"), annotated_image)

    # I'm expected points to be numbered left-to-right, top-to-bottom.
    assert image_points[1][0] > image_points[0][0]
    assert image_points[13][1] > image_points[0][1]

    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    ids, object_points, image_points = detector.get_points(rotated_image)
    assert ids.shape[0] == expected_number_of_points
    assert object_points.shape[0] == expected_number_of_points
    assert image_points.shape[0] == expected_number_of_points
    annotated_image = pdu.get_annotated_image(rotated_image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_chessboard_black_corners_10x8_rotated_90.png"), annotated_image)

    # I expected points to be numbered top-to-bottom, right-to-left
    # i.e.
    # assert image_points[1][1] > image_points[0][1]
    # assert image_points[13][0] < image_points[0][0]
    # (as above test), but this doesn't work for this image.
    # For this image, that's got all black corners, numbering
    # starts bottom-to-top and right-to-left.
    assert image_points[1][1] < image_points[0][1]
    assert image_points[13][0] > image_points[0][0]

    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    ids, object_points, image_points = detector.get_points(rotated_image)
    assert ids.shape[0] == expected_number_of_points
    assert object_points.shape[0] == expected_number_of_points
    assert image_points.shape[0] == expected_number_of_points
    annotated_image = pdu.get_annotated_image(rotated_image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_chessboard_black_corners_10x8_rotated_180.png"), annotated_image)

    # I expected points to be numbered right-to-left, bottom-to-top
    # (i.e. upside down), as we rotated by 180 degrees
    # But clearly, it won't be as the image is symmetrical.
    # So, this should be the same as the original image.
    # So, for symmetric image, points should be numbered left-to-right, top-to-bottom.
    assert image_points[1][0] > image_points[0][0]
    assert image_points[13][1] > image_points[0][1]

    # The moral of this story, is not to use a symmetric chesssboard with all black corners.


def test_non_chessboard_image():
    # Doesn't contain a chessboard, so simply checking
    # that we don't fail, and return empty arrays rather than None.
    image = cv2.imread('tests/data/processing/j_eroded.png')
    detector = ChessboardPointDetector((13, 10), 3)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 0
    assert object_points.shape[0] == 0
    assert image_points.shape[0] == 0


def test_wrong_chessboard_image():
    # Contains a chessboard, but not the right size.
    # Image comes from OpenCV online tutorials.
    image = cv2.imread('tests/data/calib-opencv/left01.jpg')
    detector = ChessboardPointDetector((13, 10), 3)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 0
    assert object_points.shape[0] == 0
    assert image_points.shape[0] == 0
