# coding=utf-8

"""
Tests for ChArUco + Chessboard implementation of PointDetector.
"""
import os
import cv2
import pytest
import numpy as np
import sksurgeryimage.calibration.charuco_plus_chessboard_point_detector as cpcbd
import sksurgeryimage.calibration.point_detector_utils as pdu
import sksurgeryimage.calibration.charuco as ch

def _create_default_detector():
    # Tutorial-section1-start
    input_image_file = "tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_9x14.png"
    input_image = cv2.imread(input_image_file)

    min_points_to_detect = 50
    num_squares = [19, 26]
    square_size_mm = [5, 4]
    chessboard_squares = [9, 14]
    chessboard_square_size_mm = 3

    point_detector = \
        cpcbd.CharucoPlusChessboardPointDetector(
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
            minimum_number_of_points=min_points_to_detect,
            number_of_charuco_squares=num_squares,
            size_of_charuco_squares=square_size_mm,
            number_of_chessboard_squares=chessboard_squares,
            chessboard_square_size=chessboard_square_size_mm,
            legacy_pattern = True # As pattern generated with OpenCV pre-4.6
        )
    # Tutorial-section1-end
    return input_image, point_detector

def test_charuco_plus_chessboard_tutorial_stuff():
    """ Code that is used in tutorials. """
    input_image, point_detector = _create_default_detector()

    #Tutorial-section2-start
    ids, object_points, image_points = point_detector.get_points(input_image)
    #Tutorial-section2-end

    #Tutorial-section3-start
    annotated_image = pdu.get_annotated_image(input_image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_plus_chessboard_tutorial_stuff.png"), annotated_image)
    #Tutorial-section3-end

    # For the above image, we should assert stuff to test it's working!
    # We expect ChArUco to start in top left corner, x-axis right, y-axis down.
    assert ids.shape[0] == 468
    assert image_points[0][0] < image_points[1][0]
    assert np.allclose(image_points[0][1], image_points[1][1], atol=0.5)
    assert np.allclose(image_points[0][0], image_points[18][0], atol=0.5)
    assert image_points[0][1] < image_points[18][1]
    # Should also check that Chessboard is in the right place.
    # We expect chessboard to start at id=364, origin bottom right, x-axis left, y-axis up.
    assert image_points[364][0] > image_points[365][0]
    assert np.allclose(image_points[364][1], image_points[365][1], atol=0.5)
    assert np.allclose(image_points[364][0], image_points[372][0], atol=0.5)
    assert image_points[364][1] > image_points[18][0]

    #Tutorial-section4-start
    reference_image = point_detector.get_reference_image()
    ids, object_points, image_points = point_detector.get_points(reference_image)
    annotated_image = pdu.get_annotated_image(reference_image, ids, image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_plus_chessboard_tutorial_stuff_reference.png"), annotated_image)
    # Tutorial-section4-end

def test_charuco_plus_chessboard_rotate_90():
    input_image, point_detector = _create_default_detector()
    ids, object_points, image_points = point_detector.get_points(input_image)
    rotated_image = cv2.rotate(input_image, cv2.ROTATE_90_CLOCKWISE)
    rotated_ids, rotated_object_points, rotated_image_points = point_detector.get_points(rotated_image)
    annotated_image_cb = pdu.get_annotated_image(rotated_image, rotated_ids, rotated_image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_plus_chessboard_rotate_90.png"), annotated_image_cb)

    # Object points should not change.
    # They are derived from the internal reference image, and keyed by id.
    assert np.allclose(object_points, rotated_object_points)

    # We expect ChArUco to start in top right corner, x-axis down, y-axis left.
    assert ids.shape[0] == 468
    assert np.allclose(rotated_image_points[0][0], rotated_image_points[1][0], atol=0.5)
    assert rotated_image_points[0][1] < rotated_image_points[1][1]
    assert rotated_image_points[0][0] > rotated_image_points[18][0]
    assert np.allclose(rotated_image_points[0][1], rotated_image_points[18][1], atol=0.5)
    # We expect chessboard to start at id=364, origin bottom left, x-axis up, y-axis right.
    assert np.allclose(rotated_image_points[364][0], rotated_image_points[365][0], atol=0.5)
    assert rotated_image_points[364][1] > rotated_image_points[365][1]
    assert rotated_image_points[364][0] < rotated_image_points[372][0]
    assert np.allclose(rotated_image_points[364][1], rotated_image_points[372][1], atol=0.5)


def test_charuco_plus_chessboard_rotate_180():
    input_image, point_detector = _create_default_detector()
    ids, object_points, image_points = point_detector.get_points(input_image)
    rotated_image = cv2.rotate(input_image, cv2.ROTATE_180)
    rotated_ids, rotated_object_points, rotated_image_points = point_detector.get_points(rotated_image)
    annotated_image_cb = pdu.get_annotated_image(rotated_image, rotated_ids, rotated_image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_plus_chessboard_rotate_180.png"), annotated_image_cb)

    # Object points should not change.
    # They are derived from the internal reference image, and keyed by id.
    assert np.allclose(object_points, rotated_object_points)

    # We expect ChArUco to start in bottom right corner, x-axis up, y-axis right.
    assert ids.shape[0] == 468
    assert rotated_image_points[0][0] > rotated_image_points[1][0]
    assert np.allclose(rotated_image_points[0][1], rotated_image_points[1][1], atol=0.5)
    assert np.allclose(rotated_image_points[0][0], rotated_image_points[18][0], atol=0.5)
    assert rotated_image_points[0][1] > rotated_image_points[18][1]
    # We expect chessboard to start at id=364, origin top right, x-axis down, y-axis left.
    assert rotated_image_points[364][0] < rotated_image_points[365][0]
    assert np.allclose(rotated_image_points[364][1], rotated_image_points[365][1], atol=0.5)
    assert np.allclose(rotated_image_points[364][0], rotated_image_points[372][0], atol=0.5)
    assert rotated_image_points[364][1] < rotated_image_points[372][1]

def test_charuco_plus_chessboard_rotate_270():
    input_image, point_detector = _create_default_detector()
    ids, object_points, image_points = point_detector.get_points(input_image)
    rotated_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotated_ids, rotated_object_points, rotated_image_points = point_detector.get_points(rotated_image)
    annotated_image_cb = pdu.get_annotated_image(rotated_image, rotated_ids, rotated_image_points)
    cv2.imwrite(os.path.join("tests", "output", "test_charuco_plus_chessboard_rotate_270.png"), annotated_image_cb)

    # Object points should not change.
    # They are derived from the internal reference image, and keyed by id.
    assert np.allclose(object_points, rotated_object_points)

    # We expect ChArUco to start in bottom left corner, x-axis left, y-axis up.
    assert ids.shape[0] == 468
    assert np.allclose(rotated_image_points[0][0], rotated_image_points[1][0], atol=0.5)
    assert rotated_image_points[0][1] > rotated_image_points[1][1]
    assert rotated_image_points[0][0] < rotated_image_points[18][0]
    assert np.allclose(rotated_image_points[0][1], rotated_image_points[18][1], atol=0.5)
    # We expect chessboard to start at id=364, origin top left, x-axis right, y-axis down.
    assert np.allclose(rotated_image_points[364][0], rotated_image_points[365][0], atol=0.5)
    assert rotated_image_points[364][1] < rotated_image_points[365][1]
    assert rotated_image_points[364][0] > rotated_image_points[372][0]
    assert np.allclose(rotated_image_points[364][1], rotated_image_points[372][1], atol=0.5)


def test_charuco_generated_image(load_reference_charuco_chessboard_image):

    ref_img = load_reference_charuco_chessboard_image

    generated_image = ch.make_charuco_with_chessboard(legacy_pattern=True)
    output_file_name = 'tests/output/pattern_4x4_19x26_5_4_with_inset_9x14.png'
    cv2.imwrite(output_file_name, generated_image)
    generated_image = cv2.imread(output_file_name)

    assert np.allclose(ref_img, generated_image)

    _, detector = _create_default_detector()
    ids_portrait, object_points_portrait, image_points_portrait = detector.get_points(ref_img)
    if ids_portrait.shape[0] > 0:
        pdu.write_annotated_image(ref_img, ids_portrait, image_points_portrait, 'tests/output/pattern_4x4_19x26_5_4_with_inset_9x14_portrait.png')

    input_file_name = 'tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_9x14_landscape.png'
    image = cv2.imread(input_file_name)
    ids_landscape, object_points_landscape, image_points_landscape = detector.get_points(image)
    if ids_landscape.shape[0] > 0:
        pdu.write_annotated_image(image, ids_landscape, image_points_landscape, 'tests/output/pattern_4x4_19x26_5_4_with_inset_9x14_landscape.png')

    assert ids_portrait.shape[0] == 468
    assert object_points_portrait.shape[0] == 468
    assert image_points_portrait.shape[0] == 468
    assert ids_landscape.shape[0] == 468
    assert object_points_landscape.shape[0] == 468
    assert image_points_landscape.shape[0] == 468

    assert np.allclose(ids_portrait, ids_landscape)
    assert np.allclose(object_points_portrait, object_points_landscape)


def test_charuco_plus_chess_invalid_because_no_dictionary():

    with pytest.raises(ValueError) as excinfo:
        detector = cpcbd.CharucoPlusChessboardPointDetector(dictionary=None)
    assert str(excinfo.value) == "dictionary is None"


def test_charuco_plus_chess_invalid_because_no_chessboard_squares():

    with pytest.raises(ValueError) as excinfo:
        detector = cpcbd.CharucoPlusChessboardPointDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                                            use_chessboard_inset=True,
                                                            number_of_chessboard_squares=None)
    assert str(excinfo.value) == "You must provide the number of chessboard corners"


def test_charuco_plus_chess_invalid_because_no_chessboard_size():

    with pytest.raises(ValueError) as excinfo:
        detector = cpcbd.CharucoPlusChessboardPointDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                                            use_chessboard_inset=True,
                                                            chessboard_square_size=None)
    assert str(excinfo.value) == "You must provide the size of chessboard squares"


def test_charuco_plus_chess_invalid_because_no_id_offset():

    with pytest.raises(ValueError) as excinfo:
        detector = cpcbd.CharucoPlusChessboardPointDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                                            use_chessboard_inset=True,
                                                            chessboard_id_offset=None)
    assert str(excinfo.value) == "You must provide chessboard ID offset"


def test_charuco_plus_chess_invalid_because_id_offset_negative():

    with pytest.raises(ValueError) as excinfo:
        detector = cpcbd.CharucoPlusChessboardPointDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                                            use_chessboard_inset=True,
                                                            chessboard_id_offset=-1)
    assert str(excinfo.value) == "Chessboard ID offset must be positive."


def test_charuco_plus_chess_invalid_because_id_offset_too_small():

    # The default 26*19 grid will result in a maximum of 25*18 corners
    # from the ChArUco bit, so we must have an id_offset of at least 450.
    with pytest.raises(ValueError) as excinfo:
        detector = cpcbd.CharucoPlusChessboardPointDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                                            use_chessboard_inset=True,
                                                            chessboard_id_offset=449)

    assert str(excinfo.value) == "Chessboard ID offset must > number of ChArUco tags."

    detector = cpcbd.CharucoPlusChessboardPointDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                                        use_chessboard_inset=True,
                                                        chessboard_id_offset=450)


def test_charuco_plus_chess_invalid_because_no_chessboard_detected(load_reference_charuco_chessboard_image):

    ref_img = load_reference_charuco_chessboard_image
    roi = ref_img[0:400, 0:400, :]

    with pytest.raises(ValueError) as excinfo:
        # If user specifies in constructor that we are using a chessboard, then a chessboard must be detected.
        detector = cpcbd.CharucoPlusChessboardPointDetector(dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                                            use_chessboard_inset=True
                                                            )
        _, _, _ = detector.get_points(roi)

    assert str(excinfo.value) == "No chessboard detected."
