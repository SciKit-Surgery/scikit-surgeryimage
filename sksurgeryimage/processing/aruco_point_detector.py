# coding=utf-8

"""
ArUco implementation of PointDetector.
"""

import logging
from cv2 import aruco
import numpy as np
from sksurgeryimage.processing.point_detector import PointDetector

LOGGER = logging.getLogger(__name__)


def get_intersect(a_1, a_2, b_1, b_2):
    """
    Returns the point of intersection of the lines
    passing through a2,a1 and b2,b1.

    See https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    :param a_1: [x, y] a point on the first line
    :param a_2: [x, y] another point on the first line
    :param b_1: [x, y] a point on the second line
    :param b_2: [x, y] another point on the second line
    """
    stacked = np.vstack([a_1, a_2, b_1, b_2])
    homogenous = np.hstack((stacked, np.ones((4, 1))))
    line_1 = np.cross(homogenous[0], homogenous[1])
    line_2 = np.cross(homogenous[2], homogenous[3])
    p_x, p_y, p_z = np.cross(line_1, line_2)
    if p_z == 0:  # lines are parallel
        return float('inf'), float('inf')
    return p_x/p_z, p_y/p_z


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
        :param model: dictionary of {id : 3D point as numpy 1x3 array}
        :param scale: if you want to cv::resize the image, specify scale factors
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
        corners, ids, _ = \
            aruco.detectMarkers(image,
                                self.dictionary,
                                parameters=self.parameters)

        number_of_points = len(corners)
        image_points = np.zeros((number_of_points, 2))
        object_points = np.zeros((number_of_points, 3))

        if self.model is None:
            object_points = np.zeros((0, 3))

        if number_of_points > 0:

            for i in range(number_of_points):
                centre = get_intersect(corners[i][0][0],  # intersect diagonals
                                       corners[i][0][2],
                                       corners[i][0][1],
                                       corners[i][0][3],
                                       )
                image_points[i][0] = centre[0]
                image_points[i][1] = centre[1]

                if self.model is not None:
                    point_id = ids[i][0]
                    object_point = self.model[point_id]

                    object_points[i][0] = object_point[0][0]
                    object_points[i][1] = object_point[0][1]
                    object_points[i][2] = object_point[0][2]

            return ids, object_points, image_points

        return np.zeros((0, 1)), object_points, image_points
