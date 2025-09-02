# coding=utf-8

"""
Functions to support camera calibration using ChArUco chessboard markers.
"""
import cv2
import numpy as np


# pylint: disable=too-many-arguments
def make_charuco_board(dictionary,
                       number_of_squares,
                       size,
                       image_size,
                       legacy_pattern: bool=True,
                       start_id: int=0):
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
    :param legacy_pattern: if True, uses the original OpenCV pattern (pre-OpenCV 4.6.0).
    :return: image, board
    """
    number_in_x, number_in_y = number_of_squares
    size_of_square, size_of_tag = size
    finish_id = start_id + np.ceil(((number_in_x * number_in_y) / 2.0)).astype(int)
    ids = np.arange(start_id, finish_id)
    board = cv2.aruco.CharucoBoard((number_in_x, number_in_y),
                                   size_of_square,
                                   size_of_tag,
                                   dictionary,
                                   ids=ids
                                   )
    board.setLegacyPattern(legacy_pattern)
    image = board.generateImage(image_size, marginSize=0, borderBits=1)
    return image, board


def detect_charuco_points(dictionary: cv2.aruco.Dictionary,
                          board,
                          image,
                          camera_matrix=None,
                          distortion_coefficients=None,
                          parameters: cv2.aruco.DetectorParameters=None):
    """
    Extracts ChArUco points. If you can provide camera matrices,
    it may be more accurate.

    :param dictionary: aruco dictionary definition
    :param board: aruco board definition
    :param image: grey scale image in which to search
    :param camera_matrix: if specified, the 3x3 camera intrinsic matrix
    :param distortion_coefficients: if specified, the distortion coefficients
    :return: marker_corners, marker_ids, chessboard_corners, chessboard_ids
    """
    if parameters is None:
        parameters = cv2.aruco.DetectorParameters()
        parameters.maxErroneousBitsInBorderRate = 0.1
        parameters.perspectiveRemovePixelPerCell = 30
        parameters.perspectiveRemoveIgnoredMarginPerCell = 0.3

    # pylint: disable=unpacking-non-sequence
    marker_corners, marker_ids, _ =\
        cv2.aruco.detectMarkers(image,
                                dictionary=dictionary,
                                parameters=parameters)

    chessboard_corners = None
    chessboard_ids = None

    if marker_corners:

        # pylint: disable=unpacking-non-sequence
        _, chessboard_corners, chessboard_ids \
            = cv2.aruco.interpolateCornersCharuco(
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
    if chessboard_corners is None or chessboard_ids is None:
        return cloned

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


# pylint: disable=too-many-arguments, too-many-locals
def make_charuco_with_chessboard(
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
        charuco_squares=(19, 26),
        charuco_size=(5, 4),
        pixels_per_millimetre=10,
        chessboard_squares=(9, 14),
        chessboard_size=3,
        chessboard_border=0.7,
        legacy_pattern=True,
        start_id=0
        ):
    """
    Helper function to make an image of a calibration target combining
    ChArUco markers and a chessboard. It's up to the caller to work out
    a nice number of pixels per millimetre, so that the resultant image
    is correctly scaled.

    Defaults are as used in SmartLiver project. Not also, that we compute
    the image and coordinates in portrait, but it's normally used in landscape.

    :param dictionary: ChArUco dictionary
    :param charuco_squares: tuple of (squares in x, squares in y)
    :param charuco_size: tuple of (external size, internal tag size) in mm
    :param pixels_per_millimetre: which determines size of eventual image.
    :param chessboard_squares: tuple of (squares in x, squares in y)
    :param chessboard_size: size of chessboard squares in mm
    :param chessboard_border: border round chessboard, as fraction of square
    :param legacy_pattern: if True, uses the original OpenCV pattern (pre-OpenCV 4.6.0).
    :param start_id: id of first marker in ChArUco board
    :return: calibration image
    """
    charuco_pixels_per_square = charuco_size[0] * pixels_per_millimetre
    size_x = charuco_squares[0] * charuco_pixels_per_square
    size_y = charuco_squares[1] * charuco_pixels_per_square
    charuco_image, _ = make_charuco_board(
        dictionary,
        number_of_squares=(charuco_squares[0], charuco_squares[1]),
        size=(charuco_size[0], charuco_size[1]),
        image_size=(size_x, size_y),
        legacy_pattern=legacy_pattern,
        start_id=start_id
    )

    centre_of_image = ((size_x - 1) / 2.0, (size_y - 1) / 2.0)

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
        ((chessboard_squares[0] / 2.0) + chessboard_border) * chessboard_scale,
        ((chessboard_squares[1] / 2.0) + chessboard_border) * chessboard_scale
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

    # pylint: disable=unsupported-assignment-operation
    charuco_image[s_y:s_y + chessboard_image.shape[0],
                  s_x:s_x + chessboard_image.shape[1]] = chessboard_image

    return charuco_image
