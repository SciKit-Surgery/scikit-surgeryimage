# coding=utf-8

"""
Tests for ChArUco implementation of PointDetector.
"""
import os
import cv2
import numpy as np
import sksurgeryimage.calibration.point_detector_utils as pdu
from sksurgeryimage.calibration.charuco_point_detector import CharucoPointDetector


def test_charuco_detector():
    image = cv2.imread('tests/data/calibration/test-charuco.png')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (13, 10), (3, 2), legacy_pattern=True)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 108
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 108
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 108
    assert image_points.shape[1] == 2
    annotated_image = pdu.get_annotated_image(image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_detector.png"), annotated_image)

    model = detector.get_model_points()
    assert isinstance(model, dict)
    assert len(model.keys()) == 108
    for i in range(108):
        np.testing.assert_array_equal(model[i], object_points[i])

    # For fun. Create and save a ChArUco board image with a different start id.
    detector = CharucoPointDetector(dictionary, (13, 10), (3, 2), legacy_pattern=True, start_id=1)
    generated_image = detector.get_reference_image()
    ids, object_points, image_points = detector.get_points(generated_image)
    annotated_image = pdu.get_annotated_image(generated_image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_detector_start_at_1.png"), annotated_image)

    # These should stil hold true for new tags. i.e. same object points.
    model = detector.get_model_points()
    assert isinstance(model, dict)
    assert len(model.keys()) == 108
    for i in range(108):
        np.testing.assert_array_equal(model[i], object_points[i])


def test_charuco_detector_with_masked_image():
    image = cv2.imread('tests/data/calibration/test-charuco-blanked.png')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (13, 10), (3, 2), legacy_pattern=True)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 45
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 45
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 45
    assert image_points.shape[1] == 2


def test_charuco_detector_1():
    image_name = 'tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_13x18.png'
    image = cv2.imread(image_name)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (19, 26), (5, 4), legacy_pattern=True)
    ids, object_points, image_points = detector.get_points(image)
    pdu.write_annotated_image(image, ids, image_points, image_name)
    expected_number = 322
    assert ids.shape[0] == expected_number
    assert ids.shape[1] == 1
    assert object_points.shape[0] == expected_number
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == expected_number
    assert image_points.shape[1] == 2


def test_charuco_detector_2():
    image_name = "tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_9x14_landscape.png"
    image = cv2.imread(image_name)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (19, 26), (5, 4), legacy_pattern=True)
    ids, object_points, image_points = detector.get_points(image)
    pdu.write_annotated_image(image, ids, image_points, image_name)
    expected_number = 364
    assert ids.shape[0] == expected_number
    assert ids.shape[1] == 1
    assert object_points.shape[0] == expected_number
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == expected_number
    assert image_points.shape[1] == 2


def test_charuco_detector_rotated_90():
    image = cv2.imread('tests/data/calibration/test-charuco.png')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (13, 10), (3, 2), legacy_pattern=True)
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 108
    annotated_image = pdu.get_annotated_image(image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_detector_rotated_0.png"), annotated_image)
    assert image_points[0][0] < image_points[1][0]
    assert image_points[0][1] == image_points[1][1]
    assert image_points[0][0] == image_points[12][0]
    assert image_points[0][1] < image_points[12][1]

    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_ids, rotated_object_points, rotated_image_points = detector.get_points(rotated_image)
    assert rotated_ids.shape[0] == 108
    annotated_image = pdu.get_annotated_image(rotated_image, rotated_ids, rotated_image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_detector_rotated_90.png"), annotated_image)

    # Having rotated, the same 3 points should be in the top-right.
    # It appears that OpenCV still reads them out in the right/same order.
    assert rotated_image_points[0][0] == rotated_image_points[1][0]
    assert rotated_image_points[0][1] < rotated_image_points[1][1]
    assert rotated_image_points[0][0] > rotated_image_points[12][0]
    assert rotated_image_points[0][1] == rotated_image_points[12][1]


def test_charuco_detector_rotated_180():
    image = cv2.imread('tests/data/calibration/test-charuco.png')
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    detector = CharucoPointDetector(dictionary, (13, 10), (3, 2), legacy_pattern=True)
    rotated_ids, rotated_object_points, rotated_image_points = detector.get_points(rotated_image)
    assert rotated_ids.shape[0] == 108
    annotated_image = pdu.get_annotated_image(rotated_image, rotated_ids, rotated_image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_detector_rotated_180.png"), annotated_image)

    # Having rotated, the same 3 points should be in the bottom-right
    # It appears that OpenCV still reads them out in the right/same order.
    assert rotated_image_points[0][0] > rotated_image_points[1][0]
    assert rotated_image_points[0][1] == rotated_image_points[1][1]
    assert rotated_image_points[0][0] == rotated_image_points[12][0]
    assert rotated_image_points[0][1] > rotated_image_points[12][1]
