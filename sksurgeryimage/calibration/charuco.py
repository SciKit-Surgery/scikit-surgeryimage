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
    image = board.draw(image_size, 1, 0)
    return image, board


def filter_out_wrong_markers(marker_corners,
                             marker_ids,
                             board):
    """
    Filters out markers that were mis-labelled. For each inner corner on the
    charuco board, if both neighbouring markers are detected, look at the
    projected positions of this corner using the perspective transformations
    obtained form the two markers. If the two positions are not close (further
    than 20 pixels away), then at least one of the markers is mis-labelled but
    we won't know which one. Remove both markers.

    :param marker_corners: marker corners detected by OpenCV
    :param marker_ids: ids of markers detected
    :param board: charuco board definition
    :return: marker_corners, marker_ids
    """
    number_of_markers = len(marker_ids)

    # Calculate local homographies for each marker
    transformations = []

    for i in range(0, number_of_markers):
        marker_id = marker_ids[i][0]
        marker_obj_corners = board.objPoints[marker_id]
        marker_obj_corners_2d = marker_obj_corners[:, 0:2].astype(np.float32)
        marker_img_corners = marker_corners[i][0].astype(np.float32)

        trans = cv2.getPerspectiveTransform(marker_obj_corners_2d,
                                            marker_img_corners)

        transformations.append(trans)

    # For each charuco corner, calculate its projected positions based on
    # the closest markers' homographies
    mask = np.ones((number_of_markers, 1), dtype=bool)
    mask = mask.flatten()
    number_of_corners = board.chessboardCorners.shape[0]
    for i in range(0, number_of_corners):
        obj_point_2d = board.chessboardCorners[i, 0:2].astype(np.float32)
        obj_point_2d = np.reshape(obj_point_2d, (-1, 1, 2))
        projected_positions = []
        neighbour_markers = []

        number_of_nearest_markers = len(board.nearestMarkerIdx[i])
        assert number_of_nearest_markers == 2
        for j in range(0, number_of_nearest_markers):
            marker_id = board.ids[board.nearestMarkerIdx[i][j][0]][0]
            marker_index = -1
            for k in range(0, number_of_markers):
                if marker_ids[k][0] == marker_id:
                    marker_index = k
                    break

            if marker_index is not -1:
                # The input point array needs to be 3 dimensional!
                out = cv2.perspectiveTransform(obj_point_2d,
                                               transformations[marker_index])
                projected_positions.append(out)
                neighbour_markers.append(marker_index)

        if len(projected_positions) > 1:
            dis = np.linalg.norm(projected_positions[0]
                                 - projected_positions[1])
            if dis > 20:
                mask[neighbour_markers] = False

    marker_ids = marker_ids[mask]
    marker_corners = np.array(marker_corners)
    marker_corners = marker_corners[mask]

    return marker_corners, marker_ids


def detect_charuco_points(dictionary, board, image,
                          camera_matrix=None,
                          distortion_coefficients=None,
                          filtering=False):
    """
    Extracts ChArUco points. If you can provide camera matrices,
    it may be more accurate.

    :param dictionary: aruco dictionary definition
    :param board: aruco board definition
    :param image: grey scale image in which to search
    :param camera_matrix: if specified, the 3x3 camera intrinsic matrix
    :param distortion_coefficients: if specified, the distortion coefficients
    :param filtering: whether or not check and filter out wrongly detected
    markers
    :return: marker_corners, marker_ids, chessboard_corners, chessboard_ids
    """
    detection_parameters = aruco.DetectorParameters_create()
    detection_parameters.maxErroneousBitsInBorderRate = 0.1
    detection_parameters.perspectiveRemovePixelPerCell = 30
    detection_parameters.perspectiveRemoveIgnoredMarginPerCell = 0.3
    marker_corners, marker_ids, _ =\
        aruco.detectMarkers(image, dictionary,
                            parameters=detection_parameters)

    chessboard_corners = None
    chessboard_ids = None

    if marker_corners:

        if filtering:
            marker_corners, marker_ids = \
                filter_out_wrong_markers(marker_corners, marker_ids, board)

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
