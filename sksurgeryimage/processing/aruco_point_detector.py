# coding=utf-8

"""
ArUco implementation of PointDetector.
"""

import logging
from sksurgeryimage.processing.point_detector import PointDetector

LOGGER = logging.getLogger(__name__)


class ArucoPointDetector(PointDetector):
    """
    Class to detect ArUco points in a 2D grey scale video image.
    """
    def __init__(self):
        super(ArucoPointDetector, self).__init__()

    def _internal_get_points(self, image):
        """
        Extracts points using OpenCV's ArUco implementation.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points
        """
        raise NotImplementedError()
