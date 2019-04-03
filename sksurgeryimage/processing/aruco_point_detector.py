# coding=utf-8

"""
ArUco implementation of PointDetector.
"""

import logging
import cv2 as c
from cv2 import aruco
import numpy as np
from sksurgeryimage.processing.point_detector import PointDetector

LOGGER = logging.getLogger(__name__)


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines
    passing through a2,a1 and b2,b1.

    See https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z == 0: # lines are parallel
        return float('inf'), float('inf')
    return x/z, y/z


class ArucoPointDetector(PointDetector):
    """
    Class to detect ArUco points in a 2D grey scale video image.

    Note: For ArUco points, these don't have to be on a regular grid.
    If you provide a 'model' which is a map of id : 3D point, the
    function _internal_get_points will provide the corresponding
    3D points of those points that were detected.
    """
    def __init__(self,
                 dictionary,
                 parameters,
                 model,
                 scale=(1, 1),
                 ):
        """
        Constructs a ArucoPointDetector.

        :param dictionary: aruco dictionary
        :param parameters: aruco parameters
        :param model: map of id : 3D point as numpy 1x3 array
        :param scale: if you want to resize the image, specify scale factors
        """
        super(ArucoPointDetector, self).__init__(scale=scale)
        self.dictionary = dictionary
        self.parameters = parameters
        self.model = model

    def _internal_get_points(self, image):
        """
        Extracts points using OpenCV's ArUco implementation.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points
        """
        corners, ids, rejectedImgPoints = \
            aruco.detectMarkers(image,
                                self.dictionary,
                                parameters=self.parameters)

        image_points = np.zeros((len(corners), 2))
        object_points = np.zeros((len(corners), 3))

        if len(corners) > 0:

            for i in range(len(corners)):
                centre = get_intersect(corners[i][0][0], # intersect diagonals
                                       corners[i][0][2],
                                       corners[i][0][1],
                                       corners[i][0][3],
                                       )
                image_points[i][0] = centre[0]
                image_points[i][1] = centre[1]

            return ids, object_points, image_points

        else:
            return np.zeros((0, 1)), object_points, image_points

