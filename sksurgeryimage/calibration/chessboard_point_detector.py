# coding=utf-8

"""
Chessboard implementation of PointDetector.
"""

import logging
import copy
from typing import Tuple
import cv2
import numpy as np
from sksurgeryimage.calibration.point_detector import PointDetector

LOGGER = logging.getLogger(__name__)

# pylint: disable=too-many-arguments, too-many-instance-attributes
class ChessboardPointDetector(PointDetector):
    """
    Class to detect chessboard points in a 2D grey scale video image.
    """
    def __init__(self,
                 number_of_corners: Tuple[int, int],
                 square_size_in_mm: int,
                 scale: Tuple[float, float]=(1.0, 1.0),
                 chessboard_flags: int=cv2.CALIB_CB_ADAPTIVE_THRESH
                                       + cv2.CALIB_CB_NORMALIZE_IMAGE
                                       + cv2.CALIB_CB_FILTER_QUADS,
                 optimisation_criteria: Tuple[int, int, float]=(cv2.TERM_CRITERIA_EPS
                                                                + cv2.TERM_CRITERIA_MAX_ITER,
                                                                30,
                                                                0.001)):
        """
        Constructs a ChessboardPointDetector.

        :param number_of_corners: tuple of (number in x, number in y), number of internal corners.
        :param square_size_in_mm: physical size of chessboard squares in mm
        :param scale: if you want to resize the image, specify scale factors
        :param chessboard_flags: OpenCV flags to pass to cv2.findChessboardCorners
        :param optimisation_criteria: criteria for cv2.cornerSubPix
        """
        super().__init__(scale=scale)
        model_points = {}
        self.number_of_corners = number_of_corners
        self.number_in_x, self.number_in_y = self.number_of_corners
        self.expected_number_of_points = self.number_in_x * self.number_in_y
        self.square_size_in_mm = square_size_in_mm
        self.object_points = np.zeros((self.expected_number_of_points, 3))
        self.ids = np.zeros((self.expected_number_of_points, 1), dtype=np.int16)
        self.chessboard_flags = chessboard_flags
        self.optimisation_criteria = optimisation_criteria

        for i in range(0, self.expected_number_of_points):
            self.object_points[i][0] = (i % self.number_in_x) \
                                       * self.square_size_in_mm
            self.object_points[i][1] = (i // self.number_in_x) \
                                       * self.square_size_in_mm
            self.object_points[i][2] = 0
            self.ids[i][0] = i
            model_points[i] = self.object_points[i]
        self.model_points = model_points


    def _internal_get_points(self, image: np.ndarray, is_distorted: bool=True):
        """
        Extracts points using OpenCV's chessboard implementation.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """
        img_points = np.zeros((0, 2))

        ret, corners = cv2.findChessboardCorners(image,
                                                 self.number_of_corners,
                                                 self.chessboard_flags)

        if ret:
            img_points = cv2.cornerSubPix(image,
                                          corners,
                                          (11, 11),
                                          (-1, -1),
                                          self.optimisation_criteria
                                          )

            # If successful, we return all ids, 3D points and 2D points.
            return copy.deepcopy(self.ids), \
                   copy.deepcopy(self.object_points), \
                   img_points.squeeze()

        # If we didn't find all points, return consistent set of 'nothing'
        return np.zeros((0, 1)), np.zeros((0, 3)), img_points
