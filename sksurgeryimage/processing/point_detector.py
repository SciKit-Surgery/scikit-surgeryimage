# coding=utf-8

"""
Base class for a PointDetector.

e.g. Chessboard corners, SIFT points, Charuco points.
"""

import logging
import cv2

LOGGER = logging.getLogger(__name__)


class PointDetector:
    """
    Class to detect points in a 2D video image.
    """
    def __init__(self):
        pass

    def get_points(self, image):
        """
        Client's call this method to extract points from an image.

        :param image: numpy 2D RGB image.
        :return: Nx2 ndarray of points
        """

        # In future, we could add methods here to accommodate:
        # 1. Caching
        # 2. Rescaling the image
        # for now, we just convert to grey scale
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.__internal_get_points(grey)

    def __internal_get_points(self, image):
        """
        Derived classes override this one.

        :param image: numpy 2D grey scale image.
        :return: Nx1 array of ids, Nx2 ndarray of points
        """
        raise RuntimeError('Derived classes should implement this method')

