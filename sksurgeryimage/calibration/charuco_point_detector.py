# coding=utf-8

"""
ChArUco implementation of PointDetector.
"""

import copy
import logging
import numpy as np
from sksurgeryimage.calibration.point_detector import PointDetector
import sksurgeryimage.calibration.charuco as charuco

LOGGER = logging.getLogger(__name__)

# pylint: disable=too-many-instance-attributes


class CharucoPointDetector(PointDetector):
    """
    Class to detect ChArUco points in a 2D video image.
    """
    def __init__(self, dictionary,
                 number_of_squares,
                 size,
                 scale=(1, 1),
                 camera_matrix=None,
                 distortion_coefficients=None,
                 filtering=False):
        """
        Constructs a CharucoPointDetector.

        :param dictionary: aruco dictionary
        :param number_of_squares: tuple of (number in x, number in y)
        :param size: tuple of (size of squares, size of internal tag) in mm
        :param scale: if you want to resize the image, specify scale factors
        :param camera_matrix: OpenCV 3x3 camera calibration matrix
        :param distortion_coefficients: OpenCV distortion coefficients
        """
        super(CharucoPointDetector, self).__init__(scale=scale)

        self.dictionary = dictionary
        self.number_of_squares = number_of_squares
        self.number_in_x = self.number_of_squares[0] - 1
        self.number_in_y = self.number_of_squares[1] - 1
        self.total_number_of_points = self.number_in_x * self.number_in_y
        self.size = size
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.filtering = filtering

        self.image, self.board = \
            charuco.make_charuco_board(self.dictionary,
                                       self.number_of_squares,
                                       self.size,
                                       (self.number_of_squares[0] * 100,
                                        self.number_of_squares[1] * 100)
                                       )
        self.object_points = np.zeros((self.total_number_of_points, 3))
        for i in range(0, self.total_number_of_points):
            self.object_points[i][0] = (i % self.number_in_x + 1) \
                                       * self.size[0]
            self.object_points[i][1] = (i // self.number_in_x + 1) \
                                       * self.size[0]
            self.object_points[i][2] = 0

    def _internal_get_points(self, image, is_distorted=True):
        """
        Extracts points using OpenCV's ChArUco implementation.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """
        img_points = np.zeros((0, 2))
        obj_points = np.zeros((0, 3))
        ids = np.zeros((0, 1))

        _, \
        _, \
        chessboard_corners, \
        chessboard_ids = \
            charuco.detect_charuco_points(self.dictionary,
                                          self.board,
                                          image,
                                          self.camera_matrix,
                                          self.distortion_coefficients,
                                          self.filtering)

        if chessboard_ids is not None \
                and chessboard_ids is not None \
                and len(chessboard_corners) > 0 \
                and len(chessboard_ids) > 0:

            ids = chessboard_ids
            obj_points = \
                np.take(self.object_points, chessboard_ids, axis=0)\
                    .reshape((-1, 3))
            img_points = chessboard_corners.reshape((-1, 2))

        return ids, obj_points, img_points

    def get_model_points(self):
        """
        Returns a [Nx3] numpy ndarray representing the model points in 3D.
        """
        return copy.deepcopy(self.object_points)
