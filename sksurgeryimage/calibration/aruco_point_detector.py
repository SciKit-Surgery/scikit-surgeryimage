# coding=utf-8

"""
ArUco implementation of PointDetector.
"""
import copy
import logging
from typing import Tuple
import cv2
import numpy as np
from sksurgeryimage.calibration.point_detector import PointDetector

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
    You must provide a 'model' which is a map of id : 3D point, the
    function _internal_get_points will provide the corresponding
    3D points of those points that were detected.
    """
    def __init__(self,
                 dictionary: cv2.aruco.Dictionary,
                 parameters: cv2.aruco.DetectorParameters,
                 model: dict,
                 scale:Tuple[float, float]=(1.0, 1.0),
                 ):
        """
        Constructs a ArucoPointDetector.

        :param dictionary: aruco dictionary
        :param parameters: aruco parameters
        :param model: dictionary of {id : 3D point as numpy 1x3 array}
        :param scale: if you want to cv::resize the image, specify scale factors
        """
        super().__init__(scale=scale)
        if dictionary is None:
            raise ValueError("dictionary is None. Programming bug.")
        if parameters is None:
            raise ValueError("parameters is None. Programming bug.")
        if model is None:
            raise ValueError("model is None. Programming bug.")
        self.dictionary = dictionary
        self.parameters = parameters
        self.model = model

    def _internal_get_points(self,
                             image: np.ndarray,
                             is_distorted: bool=True):
        """
        Extracts points using OpenCV's ArUco implementation.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points
        """
        # pylint: disable=unpacking-non-sequence
        corners, ids, _ = \
            cv2.aruco.detectMarkers(image,
                                    self.dictionary,
                                    parameters=self.parameters)

        # Check how many points we detected, whose id is in the model.
        number_of_points = 0
        for i in range(ids.shape[0]):
            if ids[i][0] in self.model:
                number_of_points += 1

        returned_ids = np.zeros((number_of_points, 1), dtype=np.int32)
        image_points = np.zeros((number_of_points, 2))
        object_points = np.zeros((number_of_points, 3))

        if number_of_points > 0:

            for i in range(number_of_points):
                if ids[i][0] in self.model.keys(): # avoids key error
                    point_id = ids[i][0]
                    object_point = self.model[point_id]

                    # intersect diagonals, more accurate than each corner.
                    centre = get_intersect(corners[i][0][0],
                                           corners[i][0][2],
                                           corners[i][0][1],
                                           corners[i][0][3],
                                           )
                    image_points[i][0] = centre[0]
                    image_points[i][1] = centre[1]

                    object_points[i][0] = object_point[0][0]
                    object_points[i][1] = object_point[0][1]
                    object_points[i][2] = object_point[0][2]
                    returned_ids[i][0] = point_id

        return returned_ids, object_points, image_points


    def get_model_points(self):
        """
        The PointDetector should return a Dictionary of id:3D point as int:np.ndarray(1,3).
        Normally, all points are planar, e.g. chessboard, so z=0. But you could
        have calibration point in 3D, so we return a 3D point. e.g. ArUco points
        on a non-planar surface.

        :return: dict[int, np.ndarray(1, 3)]
        """
        return copy.deepcopy(self.model)
