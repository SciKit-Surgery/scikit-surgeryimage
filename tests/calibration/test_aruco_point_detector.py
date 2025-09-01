# coding=utf-8

"""
Tests for Aruco implementation of PointDetector.
"""
import os
import cv2
import numpy as np
import pytest
import sksurgeryimage.calibration.point_detector_utils as pdu
import sksurgeryimage.calibration.aruco_point_detector as apd


def _get_model_1():
    model = {}
    model[860] = np.array([[0, 0, 0]], dtype=np.float32)
    model[759] = np.array([[20, 0, 0]], dtype=np.float32)
    model[752] = np.array([[40, 0, 0]], dtype=np.float32)
    model[892] = np.array([[0, 20, 0]], dtype=np.float32)
    model[304] = np.array([[20, 20, 0]], dtype=np.float32)
    model[996] = np.array([[40, 20, 0]], dtype=np.float32)
    model[11] = np.array([[0, 40, 0]], dtype=np.float32)
    model[1000] = np.array([[20, 40, 0]], dtype=np.float32)
    model[962] = np.array([[40, 40, 0]], dtype=np.float32)
    model[308] = np.array([[0, 60, 0]], dtype=np.float32)
    model[109] = np.array([[20, 60, 0]], dtype=np.float32)
    model[560] = np.array([[40, 60, 0]], dtype=np.float32)
    return model


def test_init_no_dictionary():
    with pytest.raises(ValueError) as excinfo:
        _ = apd.ArucoPointDetector(None, None, None)
    assert 'dictionary is None' in str(excinfo.value)


def test_init_no_parameters():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    with pytest.raises(ValueError) as excinfo:
        _ = apd.ArucoPointDetector(dictionary, None, None)
    assert 'parameters is None' in str(excinfo.value)


def test_init_no_model():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    with pytest.raises(ValueError) as excinfo:
        _ = apd.ArucoPointDetector(dictionary, parameters, None)
    assert 'model_points is None' in str(excinfo.value)


def test_aruco_detector_with_model():
    image_file = 'tests/data/calibration/test-aruco.png'
    model = _get_model_1()
    image = cv2.imread(image_file)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = apd.ArucoPointDetector(dictionary, parameters, model, (1, 1))
    ids, object_points, image_points = detector.get_points(image)
    annotated_image = pdu.get_annotated_image(image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_aruco_detector_with_model.png"), annotated_image)
    assert ids.shape[0] == 12
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 12
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 12
    assert image_points.shape[1] == 2
    model = detector.get_model_points()
    assert isinstance(model, dict)
    assert len(model.keys()) == 12

    # Ordering is not obvious. However, the order of the ids
    # should match the order of the image point and object point.
    # But we don't know which tag comes out first.
    # So, I've manually worked out the index of a few points.
    assert ids[11] == 860
    assert object_points[11][0] == 0
    assert object_points[11][1] == 0
    assert object_points[11][2] == 0
    assert ids[9] == 759
    assert object_points[9][0] == 20
    assert object_points[9][1] == 0
    assert object_points[9][2] == 0
    assert ids[8] == 892
    assert object_points[8][0] == 0
    assert object_points[8][1] == 20
    assert object_points[8][2] == 0

    # So, the above 3 points, 860, 759, 892 are the top-left 3 points in the model.
    # i.e. like 11 is origin, 9 is along +x axis, to the right,
    # and 8 is along +y axis, downwards.
    assert image_points[11][0] < image_points[9][0]
    assert image_points[11][1] == image_points[9][1]
    assert image_points[11][0] == image_points[8][0]
    assert image_points[11][1] < image_points[8][1]


def test_aruco_detector_with_point_not_in_model():
    image = cv2.imread('tests/data/calibration/test-aruco.png')
    model = _get_model_1()
    del model[1000]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = apd.ArucoPointDetector(dictionary, parameters, model, (1, 1))
    ids, object_points, image_points = detector.get_points(image)
    assert ids.shape[0] == 11
    assert ids.shape[1] == 1
    assert object_points.shape[0] == 11
    assert object_points.shape[1] == 3
    assert image_points.shape[0] == 11
    assert image_points.shape[1] == 2


def test_aruco_detector_with_rotate_90():
    image = cv2.imread('tests/data/calibration/test-aruco.png')
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    model = _get_model_1()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = apd.ArucoPointDetector(dictionary, parameters, model, (1, 1))
    ids, object_points, image_points = detector.get_points(rotated_image)
    annotated_image = pdu.get_annotated_image(rotated_image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_aruco_detector_rotated_90.png"), annotated_image)

    assert ids.shape[0] == 12
    assert object_points.shape[0] == 12
    assert image_points.shape[0] == 12

    assert ids[11] == 860
    assert object_points[11][0] == 0
    assert object_points[11][1] == 0
    assert object_points[11][2] == 0
    assert ids[6] == 759
    assert object_points[6][0] == 20
    assert object_points[6][1] == 0
    assert object_points[6][2] == 0
    assert ids[7] == 892
    assert object_points[7][0] == 0
    assert object_points[7][1] == 20
    assert object_points[7][2] == 0

    # So, the above 3 points, 860, 759, 892 are now the top-right 3 points in the image.
    # So, if 860 was rotated by 90 clockwise, it is now top-right.
    # Then, 759, is the +x axis in model space, but is now below the origin in image space.
    # And 892, is the +y axis in model space, but is now to the left of the origin in image space.
    assert image_points[11][0] == image_points[6][0]
    assert image_points[11][1] < image_points[6][1]
    assert image_points[11][0] > image_points[7][0]
    assert image_points[11][1] == image_points[7][1]


def test_aruco_detector_with_rotate_180():
    image = cv2.imread('tests/data/calibration/test-aruco.png')
    rotated_image = cv2.rotate(image, cv2.ROTATE_180)
    model = _get_model_1()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters()
    detector = apd.ArucoPointDetector(dictionary, parameters, model, (1, 1))
    ids, object_points, image_points = detector.get_points(rotated_image)
    annotated_image = pdu.get_annotated_image(rotated_image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_aruco_detector_rotated_180.png"), annotated_image)

    assert ids.shape[0] == 12
    assert object_points.shape[0] == 12
    assert image_points.shape[0] == 12

    assert ids[10] == 860
    assert object_points[10][0] == 0
    assert object_points[10][1] == 0
    assert object_points[10][2] == 0
    assert ids[3] == 759
    assert object_points[3][0] == 20
    assert object_points[3][1] == 0
    assert object_points[3][2] == 0
    assert ids[4] == 892
    assert object_points[4][0] == 0
    assert object_points[4][1] == 20
    assert object_points[4][2] == 0

    # So, the above 3 points, 860, 759, 892 are now the bottom-right 3 points in the image.
    # So, if the origin 860 was rotated by 180 clockwise, it is now bottom-right.
    # Then, 759, is the +x axis in model space, but is now to the left of the origin in image space.
    # And 892, is the +y axis in model space, but is now above the origin in image space.
    assert image_points[10][0] > image_points[3][0]
    assert image_points[10][1] == image_points[3][1]
    assert image_points[10][0] == image_points[4][0]
    assert image_points[10][1] > image_points[4][1]
