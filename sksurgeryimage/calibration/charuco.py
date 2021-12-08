# coding=utf-8

"""
Functions to support camera calibration using ChArUco chessboard markers.
"""
import cv2
from cv2 import aruco  # pylint: disable=no-name-in-module
import numpy as np


def make_charuco_board(dictionary, number_of_squares, size, image_size):
    """
    Generates a ChArUco pattern.

    Don't forget to select an image size that is a nice multiple of
    the square size in millimetres, to avoid any interpolation artefacts.
    You should check the resultant image has only 2 values, [0|255], and
    nothing interpolated between these two numbers.

    :param dictionary: aruco dictionary definition
    :param number_of_squares: tuple of (number in x, number in y)
    :param size: tuple of (size of chessboard square, size of internal tag), mm.
    :param image_size: tuple of (image width, image height), pixels.
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
    ChArUco board, if both neighbouring markers are detected, look at the
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
            try:
                marker_id = board.ids[board.nearestMarkerIdx[i][j][0]][0]
            except IndexError:
                marker_id = board.ids[board.nearestMarkerIdx[i][j]]

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
    :param filtering: if True, filter out wrongly detected markers
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


def make_charuco_with_chessboard(
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
        charuco_squares=(19, 26),
        charuco_size=(5, 4),
        pixels_per_millimetre=10,
        chessboard_squares=(9, 14),
        chessboard_size=3,
        chessboard_border=0.7
        ):
    """
    Helper function to make an image of a calibration target combining
    ChArUco markers and a chessboard. It's up to the caller to work out
    a nice number of pixels per millimetre, so that the resultant image
    is correctly scaled.

    Defaults are as used in SmartLiver project. Not also, that we compute
    the image and coordinates in portrait, but it's used in landscape.

    :param dictionary: ChArUco dictionary
    :param charuco_squares: tuple of (squares in x, squares in y)
    :param charuco_size: tuple of (external size, internal tag size) in mm
    :param pixels_per_millimetre: which determines size of eventual image.
    :param chessboard_squares: tuple of (squares in x, squares in y)
    :param chessboard_size: size of chessboard squares in mm
    :param chessboard_border: border round chessboard, as fraction of square
    :return: calibration image
    """
    charuco_pixels_per_square = charuco_size[0] * pixels_per_millimetre

    charuco_image, _ = make_charuco_board(
        dictionary,
        (charuco_squares[0], charuco_squares[1]),
        (charuco_size[0], charuco_size[1]),
        (charuco_squares[0] * charuco_pixels_per_square,
         charuco_squares[1] * charuco_pixels_per_square)
    )

    centre_of_image = (
        ((charuco_squares[0] * charuco_pixels_per_square) - 1) / 2.0,
        ((charuco_squares[1] * charuco_pixels_per_square) - 1) / 2.0
    )

    # Creates minimum size chessboard, one pixel per chessboard square.
    chessboard_image = np.zeros((chessboard_squares[1],
                                 chessboard_squares[0]), dtype=np.uint8)

    # pylint: disable=invalid-name
    for x in range(0, chessboard_squares[0]):
        for y in range(0, chessboard_squares[1]):
            if x % 2 == 0 and y % 2 == 0 or x % 2 == 1 and y % 2 == 1:
                chessboard_image[y][x] = 255

    # Now we rescale it up, to the right number of pixels.
    chessboard_scale = chessboard_size * pixels_per_millimetre
    chessboard_image = cv2.resize(chessboard_image,
                                  None,
                                  fx=chessboard_scale,
                                  fy=chessboard_scale,
                                  interpolation=cv2.INTER_NEAREST)

    # Draw white polygon around chessboard
    corners = np.zeros((4, 2))
    corner_offsets = (
        ((chessboard_squares[0] / 2) + chessboard_border) * chessboard_scale,
        ((chessboard_squares[1] / 2) + chessboard_border) * chessboard_scale
    )
    corners[0][0] = int(centre_of_image[0] - corner_offsets[0])
    corners[0][1] = int(centre_of_image[1] - corner_offsets[1])
    corners[1][0] = int(centre_of_image[0] + corner_offsets[0])
    corners[1][1] = int(centre_of_image[1] - corner_offsets[1])
    corners[2][0] = int(centre_of_image[0] + corner_offsets[0])
    corners[2][1] = int(centre_of_image[1] + corner_offsets[1])
    corners[3][0] = int(centre_of_image[0] - corner_offsets[0])
    corners[3][1] = int(centre_of_image[1] + corner_offsets[1])
    cv2.fillConvexPoly(charuco_image, corners.astype(np.int32), 255)

    # Insert chessboard inside white polygon
    s_x = int(centre_of_image[0] - ((chessboard_image.shape[1] - 1) / 2))
    s_y = int(centre_of_image[1] - ((chessboard_image.shape[0] - 1) / 2))

    charuco_image[s_y:s_y + chessboard_image.shape[0],
                  s_x:s_x + chessboard_image.shape[1]] = chessboard_image

    return charuco_image
