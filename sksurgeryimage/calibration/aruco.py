# coding=utf-8

"""
Functions to support camera calibration using ChArUco chessboard markers.
"""
import cv2
from cv2 import aruco  # pylint: disable=no-name-in-module
import numpy as np


def make_charuco_board(dictionary, number_of_squares, size, image_size):
    """
    Generates a ChArUco chessboard pattern.

    :param dictionary: aruco dictionary definition
    :param number_of_squares: tuple of (number in x, number in y)
    :param size: tuple of (size of chessboard square, size of internal tag)
    :param image_size: tuple of (image width, image height)
    :return: image, board
    """
    number_in_x, number_in_y = number_of_squares
    size_of_square, size_of_tag = size

    board = aruco.CharucoBoard_create(number_in_x,
                                      number_in_y,
                                      size_of_square,
                                      size_of_tag,
                                      dictionary)
    image = board.draw(image_size, 100, 1)
    return image, board


def detect_charuco_points(dictionary, board, image,
                          camera_matrix=None,
                          distortion_coefficients=None):
    """
    Extract's ChArUco points. If you can provide camera matrices,
    it may be more accurate.

    :param dictionary: aruco dictionary definition
    :param board: aruco board definition
    :param image: grey scale image in which to search
    :param camera_matrix: if specified, the 3x3 camera intrinsic matrix
    :param distortion_coefficients: if specified, the distortion coefficients
    :return: marker_corners, marker_ids, chessboard_corners, chessboard_ids
    """
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary)

    chessboard_corners = None
    chessboard_ids = None

    if marker_corners:

        _, chessboard_corners, chessboard_ids \
            = aruco.interpolateCornersCharuco(
                markerCorners=marker_corners,
                markerIds=marker_ids,
                image=image,
                board=board,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion_coefficients
                )

    return marker_corners,\
        marker_ids,\
        chessboard_corners,\
        chessboard_ids


def draw_charuco_corners(image, chessboard_corners, chessboard_ids):
    """
    Function to draw chessboard corners on an image.

    :param image: input image
    :param chessboard_corners: from detect_charuco_points
    :param chessboard_ids: from detect_charuco_points
    :return: new image with corners marked
    """
    cloned = image.copy()

    output = cv2.aruco.drawDetectedCornersCharuco(
        image=cloned,
        charucoCorners=chessboard_corners,
        charucoIds=chessboard_ids)

    return output


def erase_charuco_markers(image, marker_corners):
    """
    Method to automatically blank out ChArUco markers,
    leaving an image that looks like it contains just a
    chessboard, rather than ChArUco board. It does
    this by drawing a plain white polygon, with vertices
    defined by the tag detection process. So, on a synthetic
    image, this works perfectly. On a real image, due to blurring
    or other artefacts such as combing, there may be some residual.

    :param image: image containing a view of a ChArUco board.
    :param marker_corners: detected corners
    :return: edited image
    """
    cloned = image.copy()
    for marker in marker_corners:
        reshaped = marker.reshape(-1, 1, 2).astype(int)
        asarray = np.asarray(reshaped)
        cv2.fillConvexPoly(cloned, asarray, 255)
    return cloned
