# coding=utf-8

"""
ChArUco implementation of PointDetector.
"""

import logging
import cv2

LOGGER = logging.getLogger(__name__)


class CharucoPointDetector:
    """
    Class to detect ChArUco points in a 2D grey scale video image.
    """
    def __init__(self):
        super(CharucoPointDetector, self).__init__()

    def __internal_get_points(self, image):
        """
        Extracts points using OpenCV's ChArUco implementation.

        :param image: numpy 2D grey scale image.
        :return: Nx1 array of ids, Nx2 ndarray of points
        """
        pass

