# coding=utf-8

"""
ArUco implementation of PointDetector.
"""

import logging
import cv2

LOGGER = logging.getLogger(__name__)


class ArucoPointDetector:
    """
    Class to detect ArUco points in a 2D grey scale video image.
    """
    def __init__(self):
        super(ArucoPointDetector, self).__init__()

    def __internal_get_points(self, image):
        """
        Extracts points using OpenCV's ArUco implementation.

        :param image: numpy 2D grey scale image.
        :return: Nx1 array of ids, Nx2 ndarray of points
        """
        pass

