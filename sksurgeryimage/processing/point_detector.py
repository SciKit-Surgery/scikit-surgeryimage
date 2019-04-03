# coding=utf-8

"""
Base class for a PointDetector.

e.g. Chessboard corners, SIFT points, Charuco points etc.
"""

import logging
import numpy as np
import cv2

LOGGER = logging.getLogger(__name__)

#pylint:disable=useless-object-inheritance


class PointDetector(object):
    """
    Class to detect points in a 2D video image.
    :param scale: tuple (x scale, y scale) to scale up/down the image
    """
    def __init__(self, scale=(1, 1)):

        self.scale = scale
        self.scale_x, self.scale_y = scale

    def get_points(self, image):
        """
        Client's call this method to extract points from an image.

        :param image: numpy 2D RGB image.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """

        if image is None:
            raise TypeError('image is None')
        if not isinstance(image, np.ndarray):
            raise TypeError('image is not an ndarray')
        if image.shape[2] != 3:
            raise ValueError("image doesn't have 3 channels")

        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        is_resized = False
        if self.scale_x != 1 or self.scale_y != 1:
            resized = cv2.resize(grey, None, fx=self.scale_x, fy=self.scale_y,
                                 interpolation=cv2.INTER_LINEAR)
            is_resized = True
        else:
            resized = grey

        ids, object_points, image_points = self._internal_get_points(resized)

        if is_resized:
            image_points[:, 0] /= self.scale_x
            image_points[:, 1] /= self.scale_y

        return ids, object_points, image_points

    def _internal_get_points(self, image):
        """
        Derived classes override this one.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """
        raise NotImplementedError()
