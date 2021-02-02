# coding=utf-8

"""
ChArUco + Chessboard implementation of PointDetector.
"""

# pylint: disable=too-many-instance-attributes

import copy
import logging
import numpy as np
import cv2
from sksurgeryimage.calibration.point_detector import PointDetector
import sksurgeryimage.calibration.charuco_point_detector as cpd
import sksurgeryimage.calibration.chessboard_point_detector as cbpd

LOGGER = logging.getLogger(__name__)


class CharucoPlusChessboardPointDetector(PointDetector):
    """
    Class to detect ChArUco points and Chessboard points
    in a 2D grey scale video image.
    """
    def __init__(self,
                 reference_image,
                 minimum_number_of_points=50,
                 scale=(1, 1),
                 number_of_charuco_squares=(19, 26),
                 size_of_charuco_squares=(5, 4),
                 dictionary=cv2.aruco.getPredefinedDictionary(
                     cv2.aruco.DICT_4X4_250),
                 camera_matrix=None,
                 distortion_coeff=None,
                 charuco_filtering=False,
                 use_chessboard_inset=True,
                 number_of_chessboard_squares=(9, 14),
                 chessboard_square_size=3,
                 chessboard_id_offset=500,
                 error_if_no_chessboard=True,
                 error_if_no_charuco=False,
                 ):
        """
        Constructs a CharucoPlusChessboardPointDetector.

        :param reference_image: mandatory example of image with all tags on
        :param dictionary: aruco dictionary
        :param number_of_charuco_squares: tuple of (number in x, number in y)
        :param size_of_charuco_squares: tuple of size (external, internal) in mm
        :param minimum_number_of_points: combined minimum number of points
        :param scale: if you want to resize the image, specify scale factors
        :param use_chessboard_inset: True if we want to use a chessboard inset
        :param number_of_chessboard_squares: tuple of (num in x, num in y)
        :param chessboard_square_size: size in millimetres of chessboard squares
        :param chessboard_id_offset: offset to add to chessboard IDs.
        :param error_if_no_chessboard: if True, throws Exception when
        no chessboard is seen
        :param error_if_no_charuco: if True, throws Exception when
        no ChArUco tags are seen
        """
        super(CharucoPlusChessboardPointDetector, self).__init__(scale=scale)

        if reference_image is None:
            raise ValueError("You must provide a reference image of all points")
        if not isinstance(reference_image, np.ndarray):
            raise ValueError("The reference image must be a numpy ndarray")

        self.number_of_charuco_squares = number_of_charuco_squares
        self.size_of_charuco_squares = size_of_charuco_squares
        self.minimum_number_of_points = minimum_number_of_points
        self.charuco_filtering = charuco_filtering
        self.number_of_chessboard_squares = number_of_chessboard_squares
        self.chessboard_square_size = chessboard_square_size
        self.chessboard_id_offset = chessboard_id_offset
        self.error_if_no_chessboard = error_if_no_chessboard
        self.error_if_no_charuco = error_if_no_charuco

        if use_chessboard_inset and not self.number_of_chessboard_squares:
            raise ValueError(
                "You must provide the number of chessboard corners")
        if use_chessboard_inset and not self.chessboard_square_size:
            raise ValueError("You must provide the size of chessboard squares")
        if use_chessboard_inset and not self.chessboard_id_offset:
            raise ValueError("You must provide chessboard ID offset")
        if use_chessboard_inset and self.chessboard_id_offset <= 0:
            raise ValueError("Chessboard ID offset must be positive.")
        if use_chessboard_inset \
                and self.chessboard_id_offset < \
                (self.number_of_charuco_squares[0] - 1) \
                * (self.number_of_charuco_squares[1] - 1):
            raise ValueError("Chessboard ID offset "
                             "must > number of ChArUco tags.")

        self.charuco_point_detector = \
            cpd.CharucoPointDetector(dictionary,
                                     self.number_of_charuco_squares,
                                     self.size_of_charuco_squares,
                                     filtering=self.charuco_filtering,
                                     camera_matrix=camera_matrix,
                                     distortion_coefficients=distortion_coeff
                                     )

        self.chessboard_point_detector = None
        self.chessboard_offset = [0] * 3

        if use_chessboard_inset:

            charucoboard_size_x = self.number_of_charuco_squares[0] * \
                self.size_of_charuco_squares[0]
            charucoboard_size_y = self.number_of_charuco_squares[1] * \
                self.size_of_charuco_squares[0]

            chessboard_size_x = self.number_of_chessboard_squares[0] \
                * self.chessboard_square_size
            chessboard_size_y = self.number_of_chessboard_squares[1] \
                * self.chessboard_square_size

            self.chessboard_offset[0] = \
                (charucoboard_size_x - chessboard_size_x) / 2 + \
                chessboard_size_x - self.chessboard_square_size

            self.chessboard_offset[1] = \
                (charucoboard_size_y - chessboard_size_y) / 2 + \
                self.chessboard_square_size

            self.chessboard_point_detector = \
                cbpd.ChessboardPointDetector(
                    (self.number_of_chessboard_squares[0] - 1,
                     self.number_of_chessboard_squares[1] - 1),
                    self.chessboard_square_size
                    )
        # Run this detector on the reference image, to get a model
        # of ALL the available points.
        _, self.model_points, _ = self.get_points(reference_image)
        if self.model_points.shape[0] == 0:
            raise ValueError("No reference model points were found")

    def _internal_get_points(self, image, is_distorted=True):
        """
        Extracts points using scikit-surgeryimage ChArUcoPointDetector
        and ChessboardPointDetector classes.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """
        charuco_ids, charuco_object_points, charuco_image_points = \
            self.charuco_point_detector.get_points(image)

        if self.error_if_no_charuco and charuco_ids.shape[0] == 0:
            raise ValueError("No ChArUco detected.")

        total_number_of_points = charuco_ids.shape[0]

        if self.chessboard_point_detector:

            chess_ids, chess_object_points, chess_image_points = \
                self.chessboard_point_detector.get_points(image)

            if self.error_if_no_chessboard and chess_ids.shape[0] == 0:
                raise ValueError("No chessboard detected.")

            total_number_of_points = total_number_of_points + \
                chess_ids.shape[0]

            # Prepare to merge charuco and chessboard points
            chess_ids = chess_ids + self.chessboard_id_offset
            chess_object_points = chess_object_points * [-1, 1, 1]\
                + self.chessboard_offset

            if charuco_ids.shape[0] == 0:

                # No merging required
                charuco_ids = chess_ids
                charuco_object_points = chess_object_points
                charuco_image_points = chess_image_points

            elif charuco_ids.shape[0] != 0 and chess_ids.shape[0] != 0:

                charuco_ids = np.append(charuco_ids,
                                        chess_ids,
                                        axis=0)
                charuco_object_points = np.append(charuco_object_points,
                                                  chess_object_points,
                                                  axis=0)
                charuco_image_points = np.append(charuco_image_points,
                                                 chess_image_points,
                                                 axis=0)

        if total_number_of_points < self.minimum_number_of_points:
            LOGGER.info("Not enough points detected. Discard.")
            return np.zeros((0, 1)), np.zeros((0, 3)), np.zeros((0, 2))

        return charuco_ids, charuco_object_points, charuco_image_points

    def get_model_points(self):
        """
        Returns a [Nx3] numpy ndarray representing the model points in 3D.
        """
        return copy.deepcopy(self.model_points)
