# coding=utf-8

"""
Chessboard implementation of PointDetector.
"""

import logging
import cv2
import numpy as np
from sksurgeryimage.processing.point_detector import PointDetector

LOGGER = logging.getLogger(__name__)


class ChessboardPointDetector(PointDetector):
    """
    Class to detect chessboard points in a 2D grey scale video image.
    """
    def __init__(self, number_of_corners, square_size_in_mm, scale=(1, 1)):
        """
        Constructs a ChessboardPointDetector.

        :param number_of_corners: tuple of (number in x, number in y)
        :param square_size_in_mm: physical size of chessboard squares in mm
        :param scale: if you want to resize the image, specify scale factors
        """
        super(ChessboardPointDetector, self).__init__(scale=scale)

        self.number_of_corners = number_of_corners
        self.number_in_x, self.number_in_y = self.number_of_corners
        self.expected_number_of_points = self.number_in_x * self.number_in_y
        self.square_size_in_mm = square_size_in_mm

        self.object_points = np.zeros((self.expected_number_of_points, 3))
        self.ids = np.zeros((self.expected_number_of_points, 1))
        for i in range(0, self.expected_number_of_points):
            self.object_points[i][0] = (i % self.number_in_x) \
                                       * self.square_size_in_mm
            self.object_points[i][1] = (i // self.number_in_x) \
                                       * self.square_size_in_mm
            self.object_points[i][2] = 0
            self.ids[i][0] = i

    def _internal_get_points(self, image):
        """
        Extracts points using OpenCV's chessboard implementation.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """
        img_points = np.zeros((0, 2))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)

        ret, corners = cv2.findChessboardCorners(image,
                                                 self.number_of_corners,
                                                 None)

        if ret:
            img_points = cv2.cornerSubPix(image,
                                          corners,
                                          (11, 11),
                                          (-1, -1),
                                          criteria
                                          )

            # If successful, we return all ids, 3D points and 2D points.
            return self.ids, self.object_points, img_points.squeeze()

        # If we didn't find all points, return consistent set of 'nothing'
        return np.zeros((0, 1)), np.zeros((0, 3)), img_points
