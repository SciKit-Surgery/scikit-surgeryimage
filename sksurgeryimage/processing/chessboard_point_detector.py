# coding=utf-8

"""
Chessboard implementation of PointDetector.
"""

import logging
import cv2

LOGGER = logging.getLogger(__name__)


class ChessboardPointDetector:
    """
    Class to detect chessboard points in a 2D grey scale video image.
    """
    def __init__(self):
        super(ChessboardPointDetector, self).__init__()

    def __internal_get_points(self, image):
        """
        Extracts points using OpenCV's chessboard implementation.

        :param image: numpy 2D grey scale image.
        :return: Nx1 array of ids, Nx2 ndarray of points
        """
        pass

