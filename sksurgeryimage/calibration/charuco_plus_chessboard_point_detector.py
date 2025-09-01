# coding=utf-8

"""
ChArUco + Chessboard implementation of PointDetector.
"""

# pylint: disable=too-many-instance-attributes

import copy
import logging
from typing import Tuple
import numpy as np
import cv2
import sksurgerycore.algorithms.procrustes as proc
import sksurgeryimage.calibration.point_detector as pd
import sksurgeryimage.calibration.charuco as ch
import sksurgeryimage.calibration.charuco_point_detector as cpd
import sksurgeryimage.calibration.chessboard_point_detector as cbpd

LOGGER = logging.getLogger(__name__)


class CharucoPlusChessboardPointDetector(pd.PointDetector):
    """
    Class to detect ChArUco points and Chessboard points
    in a 2D grey scale video image.
    """
    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
    def __init__(self,
                 dictionary: cv2.aruco.Dictionary,
                 number_of_charuco_squares=(19, 26),
                 size_of_charuco_squares=(5, 4),
                 scale=(1, 1),
                 start_id=0,
                 camera_matrix=None,
                 distortion_coeff=None,
                 use_chessboard_inset=True,
                 number_of_chessboard_squares=(9, 14),
                 chessboard_square_size=3,
                 chessboard_id_offset=500,
                 minimum_number_of_points=50,
                 error_if_no_chessboard=True,
                 error_if_no_charuco=False,
                 legacy_pattern=True,
                 parameters: cv2.aruco.DetectorParameters = None,
                 chessboard_flags: int = cv2.CALIB_CB_ADAPTIVE_THRESH
                                         + cv2.CALIB_CB_NORMALIZE_IMAGE
                                         + cv2.CALIB_CB_FILTER_QUADS,
                 optimisation_criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS
                                                                  + cv2.TERM_CRITERIA_MAX_ITER,
                                                                  30,
                                                                  0.001)
                 ):
        """
        Constructs a CharucoPlusChessboardPointDetector.

        :param dictionary: aruco dictionary
        :param number_of_charuco_squares: tuple of (number in x, number in y)
        :param size_of_charuco_squares: tuple of size (external, internal) in mm
        :param scale: if you want to resize the image, specify scale factors (see base class).
        :param start_id: id of first marker in ChArUco board
        :param camera_matrix: OpenCV 3x3 camera calibration matrix
        :param distortion_coefficients: OpenCV distortion coefficients
        :param use_chessboard_inset: True if we want to use a chessboard inset
        :param number_of_chessboard_squares: tuple of (num in x, num in y)
        :param chessboard_square_size: size in millimetres of chessboard squares
        :param chessboard_id_offset: offset to add to chessboard IDs.
        :param minimum_number_of_points: combined minimum number of points
        :param error_if_no_chessboard: if True, throws Exception when
               no chessboard is seen
        :param error_if_no_charuco: if True, throws Exception when
               no ChArUco tags are seen
        :param legacy_pattern: if True, uses OpenCV pre-4.6 ChArUco pattern
        :param parameters: OpenCV aruco DetectorParameters,
               if None, will create reasonable defaults.
        """
        super().__init__(scale=scale)
        self.number_of_charuco_squares = number_of_charuco_squares
        self.size_of_charuco_squares = size_of_charuco_squares
        self.minimum_number_of_points = minimum_number_of_points
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
                                     start_id=start_id,
                                     camera_matrix=camera_matrix,
                                     distortion_coefficients=distortion_coeff,
                                     legacy_pattern=legacy_pattern,
                                     parameters=parameters
                                     )

        self.chessboard_point_detector = None

        if use_chessboard_inset:

            self.chessboard_point_detector = \
                cbpd.ChessboardPointDetector(
                    number_of_corners=(self.number_of_chessboard_squares[0] - 1,
                                       self.number_of_chessboard_squares[1] - 1),
                    square_size_in_mm=self.chessboard_square_size,
                    chessboard_flags=chessboard_flags,
                    optimisation_criteria=optimisation_criteria
                    )

        self.reference_image = ch.make_charuco_with_chessboard(
            dictionary=dictionary,
            charuco_squares=number_of_charuco_squares,
            charuco_size=size_of_charuco_squares,
            chessboard_squares=number_of_chessboard_squares,
            chessboard_size=chessboard_square_size,
            legacy_pattern=legacy_pattern,
            start_id=start_id,
            pixels_per_millimetre=(size_of_charuco_squares[0]
                                   * size_of_charuco_squares[1])
        )

        # Need to map between chessboard coordinates and
        # ChArUco coordinates and keep them consistent.
        self.rotation_matrix = np.eye(3)
        self.translation_vector = np.zeros((3, 1))
        if use_chessboard_inset:
            _, chess_object_points, chess_image_points \
                = self.chessboard_point_detector.get_points(self.reference_image)

            # Pick 3 points, the origin, the furthest in x-axis,
            # furthest in y-axis, in chessboard coords.
            fixed_points = np.zeros((3,3))
            fixed_points[0][0] = chess_object_points[0][0]
            fixed_points[0][1] = chess_object_points[0][1]
            x_offset = number_of_chessboard_squares[0] - 2
            fixed_points[1][0] = chess_object_points[x_offset][0]
            fixed_points[1][1] = chess_object_points[x_offset][1]
            y_offset = ((number_of_chessboard_squares[0] - 1)
                        * (number_of_chessboard_squares[1] - 2))
            fixed_points[2][0] = chess_object_points[y_offset][0]
            fixed_points[2][1] = chess_object_points[y_offset][1]

            # Now we need the SAME points in ChArUco coords.
            _, charuco_object_points, charuco_image_points = (
                self.charuco_point_detector.get_points(self.reference_image))
            moving_points = np.zeros((3,3))
            charuco_origin_img = charuco_image_points[0]
            charuco_opposite_img = charuco_image_points[-1]
            charuco_origin_obj = charuco_object_points[0]
            charuco_opposite_obj = charuco_object_points[-1]

            charuco_pix_per_mm_x = ((charuco_origin_img[0] - charuco_opposite_img[0])
                                    / (charuco_origin_obj[0] - charuco_opposite_obj[0]))
            charuco_pix_per_mm_y = ((charuco_origin_img[1] - charuco_opposite_img[1])
                                    / (charuco_origin_obj[1] - charuco_opposite_obj[1]))

            moving_points[0][0] = ((chess_image_points[0][0] - charuco_origin_img[0])
                                   / charuco_pix_per_mm_x + charuco_origin_obj[0])
            moving_points[0][1] = ((chess_image_points[0][1] - charuco_origin_img[1])
                                   / charuco_pix_per_mm_y + charuco_origin_obj[1])
            moving_points[1][0] = ((chess_image_points[x_offset][0] - charuco_origin_img[0])
                                   / charuco_pix_per_mm_x + charuco_origin_obj[0])
            moving_points[1][1] = ((chess_image_points[x_offset][1] - charuco_origin_img[1])
                                   / charuco_pix_per_mm_y + charuco_origin_obj[1])
            moving_points[2][0] = ((chess_image_points[y_offset][0] - charuco_origin_img[0])
                                   / charuco_pix_per_mm_x + charuco_origin_obj[0])
            moving_points[2][1] = ((chess_image_points[y_offset][1] - charuco_origin_img[1])
                                   / charuco_pix_per_mm_y + charuco_origin_obj[1])

            # Do point-based rigid registration
            self.rotation_matrix, self.translation_vector, fre \
                = proc.orthogonal_procrustes(fixed=fixed_points, moving=moving_points)
            if fre > 0.01:
                raise ValueError(f"High fiducial registration error when "
                                 f"registering chessboard to ChArUco: {fre:.2f}mm")

        # Run this detector on the reference image, to get a model of ALL the available points.
        ids, object_points, _ = self.get_points(self.reference_image)
        model_points = {}
        for i in range(0, ids.shape[0]):
            idx = ids[i][0]
            model_points[idx] = object_points[i][0]
        self.model_points = model_points

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

            # 2025-09-01: Retesting in OpenCV 4.12.
            # With image: tests/data/calibration/pattern_4x4_19x26_5_4_with_inset_9_14.png,
            # origin is bottom right of chessboard image, with x left and y up.
            chess_ids, chess_object_points, chess_image_points = \
                self.chessboard_point_detector.get_points(image)

            if self.error_if_no_chessboard and chess_ids.shape[0] == 0:
                raise ValueError("No chessboard detected.")

            total_number_of_points = total_number_of_points + \
                chess_ids.shape[0]

            # Prepare to merge charuco and chessboard points
            chess_ids = chess_ids + self.chessboard_id_offset

            # Map chessboard points into ChArUco space
            chess_object_points = np.transpose(
                np.matmul(self.rotation_matrix, np.transpose(chess_object_points))
                + self.translation_vector
            )

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

    def get_reference_image(self):
        """
        Returns the generated ChArUco+Chessboard image.

        :return: numpy 2D grey scale image.
        """
        return copy.deepcopy(self.reference_image)
