# coding=utf-8

"""
ArUco implementation of PointDetector.
"""
import logging
from typing import Tuple
import cv2
import numpy as np
import sksurgeryimage.calibration.point_detector as pd
import sksurgeryimage.calibration.point_detector_utils as pdu


LOGGER = logging.getLogger(__name__)


class ArucoPointDetector(pd.PointDetector):
    """
    Class to detect ArUco points in a 2D grey scale video image.

    Note: For ArUco points, these don't have to be on a regular grid.
    You must provide a 'model' which is a map of id : 3D point, the
    function _internal_get_points will provide the corresponding
    3D points of those points that were detected.

    What if you want a regular grid of ArUco points? You are probably
    better off using a Charuco board, which is a chessboard with
    ArUco points inside the white squares. See CharucoPointDetector.
    """
    def __init__(self,
                 dictionary: cv2.aruco.Dictionary,
                 parameters: cv2.aruco.DetectorParameters,
                 model_points: dict,
                 scale:Tuple[float, float]=(1.0, 1.0),
                 ):
        """
        Constructs a ArucoPointDetector.

        :param dictionary: aruco dictionary
        :param parameters: aruco parameters
        :param model_points: dictionary of {id : 3D point as numpy 1x3 array}
        :param scale: if you want to cv::resize the image, specify scale factors
        """
        super().__init__(scale=scale)
        if dictionary is None:
            raise ValueError("dictionary is None. Programming bug.")
        if parameters is None:
            raise ValueError("parameters is None. Programming bug.")
        if model_points is None:
            raise ValueError("model_points is None. Programming bug.")
        self.dictionary = dictionary
        self.parameters = parameters
        self.model_points = model_points

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

        number_of_points = pdu.get_number_of_points(ids, self.model_points)

        returned_ids = np.zeros((number_of_points, 1), dtype=np.int32)
        image_points = np.zeros((number_of_points, 2))
        object_points = np.zeros((number_of_points, 3))

        if number_of_points > 0:

            for i in range(number_of_points):
                if ids[i][0] in self.model_points.keys(): # avoids key error
                    point_id = ids[i][0]
                    object_point = self.model_points[point_id]

                    # intersect diagonals, more accurate than each corner.
                    centre = pdu.get_intersect(corners[i][0][0],
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
