# coding=utf-8

"""
ChArUco implementation of PointDetector.
"""
import logging
import copy
import numpy as np
import cv2
import sksurgeryimage.calibration.point_detector_utils as pdu
from sksurgeryimage.calibration.point_detector import PointDetector
from sksurgeryimage.calibration import charuco

LOGGER = logging.getLogger(__name__)

# pylint: disable=too-many-instance-attributes, too-many-locals
class CharucoPointDetector(PointDetector):
    """
    Class to detect ChArUco points in a 2D video image.
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 dictionary: cv2.aruco.Dictionary,
                 number_of_squares,
                 size,
                 scale=(1, 1),
                 start_id=0,
                 camera_matrix=None,
                 distortion_coefficients=None,
                 legacy_pattern=True,
                 parameters: cv2.aruco.DetectorParameters=None
                 ):
        """
        Constructs a CharucoPointDetector.

        :param dictionary: aruco dictionary
        :param number_of_squares: tuple of (number in x, number in y)
        :param size: tuple of (size of squares, size of internal tag) in mm
        :param scale: if you want to resize the image, specify scale factors
        :param start_id: id of first marker in ChArUco board
        :param camera_matrix: OpenCV 3x3 camera calibration matrix
        :param distortion_coefficients: OpenCV distortion coefficients
        :param legacy_pattern: if True, uses OpenCV pre-4.6 ChArUco pattern
        :param parameters: OpenCV aruco DetectorParameters, if None,
               will create reasonable defaults.
        """
        super().__init__(scale=scale)
        if dictionary is None:
            raise ValueError("dictionary is None")
        self.dictionary = dictionary
        self.number_of_squares = number_of_squares
        self.number_in_x = self.number_of_squares[0] - 1
        self.number_in_y = self.number_of_squares[1] - 1
        self.size = size
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.parameters = parameters
        self.total_number_of_points = self.number_in_x * self.number_in_y

        self.reference_image, self.board = \
            charuco.make_charuco_board(self.dictionary,
                                       self.number_of_squares,
                                       self.size,
                                       (self.number_of_squares[0] * 100,
                                        self.number_of_squares[1] * 100),
                                       legacy_pattern=legacy_pattern,
                                       start_id=start_id
                                       )
        model_points = {}
        _, _, _, chessboard_ids = charuco.detect_charuco_points(self.dictionary,
                                                                self.board,
                                                                self.reference_image,
                                                                self.camera_matrix,
                                                                self.distortion_coefficients,
                                                                self.parameters)
        chessboard_corners_3d = self.board.getChessboardCorners()
        number_of_chessboard_ids = chessboard_ids.shape[0]
        number_of_chessboard_corners = len(chessboard_corners_3d)
        if number_of_chessboard_ids != number_of_chessboard_corners:
            raise ValueError(f"Number of chessboard ids {number_of_chessboard_ids}, "
                f"doesn't match number of chessboard corners {number_of_chessboard_ids}. ")
        for i in range(0, number_of_chessboard_ids):
            idx = chessboard_ids[i][0]
            if idx < 0 or idx >= number_of_chessboard_corners:
                raise ValueError(f"chessboard id {idx} is out of range "
                                 f"0 to {number_of_chessboard_corners-1}")
            # pylint: disable=unsubscriptable-object
            model_points[idx] = chessboard_corners_3d[idx]
        self.model_points = model_points

    def _internal_get_points(self, image, is_distorted=True):
        """
        Extracts points using OpenCV's ChArUco implementation.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """
        _, \
        _, \
        chessboard_corners, \
        chessboard_ids = \
            charuco.detect_charuco_points(self.dictionary,
                                          self.board,
                                          image,
                                          self.camera_matrix,
                                          self.distortion_coefficients)

        # Check how many points we detected, whose id is in the model.
        number_of_points = 0
        if chessboard_corners is not None \
            and chessboard_ids is not None \
            and len(chessboard_corners) > 0 \
            and len(chessboard_ids) > 0:
            number_of_points = pdu.get_number_of_points(chessboard_ids, self.model_points)

        returned_ids = np.zeros((number_of_points, 1), dtype=np.int32)
        image_points = np.zeros((number_of_points, 2))
        object_points = np.zeros((number_of_points, 3))

        if number_of_points > 0:

            for i in range(chessboard_ids.shape[0]):
                idx = chessboard_ids[i][0]
                returned_ids[i][0] = idx
                image_points[i][0] = chessboard_corners[i][0][0]
                image_points[i][1] = chessboard_corners[i][0][1]
                object_points[i][0] = self.model_points[idx][0]
                object_points[i][1] = self.model_points[idx][1]
                object_points[i][2] = self.model_points[idx][2]

        return returned_ids, object_points, image_points

    def get_reference_image(self):
        """
        Returns the generated ChArUco board image.

        :return: numpy 2D grey scale image.
        """
        return copy.deepcopy(self.reference_image)
