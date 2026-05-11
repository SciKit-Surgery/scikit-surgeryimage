# -*- coding: utf-8 -*-

"""
Dotty Grid implementation of PointDetector.
"""

# pylint:disable=too-many-instance-attributes

import logging
import cv2
import numpy as np
from sksurgeryimage.calibration.point_detector import PointDetector

LOGGER = logging.getLogger(__name__)


def create_model_points(dots_rows_columns: (int, int),
                        pixels_per_mm: int,
                        dot_separation: float) -> np.ndarray:
    """Generate the expected locations of dots in the pattern, in pixel space.

    :param dots_rows_columns: Number of rows, number of columns
    :type dots_rows_columns: [int, int]
    :param pixels_per_mm: Pixels per mm
    :type pixels_per_mm: int
    :param dot_separation: Distance between dots in mm
    :type dot_separation: float
    :return: array pf point info - [id, x_pix, y_pix, x_mm, y_mm, z_mm]
    :rtype: np.ndarray
    """

    number_of_points = dots_rows_columns[0] * dots_rows_columns[1]
    model_points = np.zeros((number_of_points, 6))
    counter = 0
    for y_index in range(dots_rows_columns[0]):
        for x_index in range(dots_rows_columns[1]):
            model_points[counter][0] = counter
            model_points[counter][1] = (x_index + 1) * pixels_per_mm
            model_points[counter][2] = (y_index + 1) * pixels_per_mm
            model_points[counter][3] = x_index * dot_separation
            model_points[counter][4] = y_index * dot_separation
            model_points[counter][5] = 0
            counter = counter + 1

    return model_points


class DottyGridPointDetector(PointDetector):
    """
    Class to detect a grid of dots in a 2D grey scale video image.

    More specifically, a grid of dots with 4 larger dots at known locations.
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 model_points,
                 list_of_indexes,
                 camera_intrinsics,
                 distortion_coefficients,
                 scale=(1, 1),
                 rms=30,
                 gaussian_sigma=5,
                 threshold_window_size=151,
                 threshold_offset=20,
                 min_area=50,
                 max_area=50000,
                 dot_detector_params=None
                 ):
        """
        Constructs a PointDetector that extracts a grid of dots,
        with 4 extra large dots, at known locations.

        Requires camera_intrinsics and distortion_coefficients to be provided,
        then these are used as a reference transform to undistort
        the image, which makes matching to a reference grid and identifying
        point indexes more reliable.

        The list of indexes, must be of length 4, and correspond to
        top-left, top-right, bottom-left, bottom-right bigger blobs.

        :param model_points: numpy ndarray of id, x_pix, y_pix, x_mm, y_mm, z_mm
        :param list_of_indexes: list of specific indexes to use as fiducials
        :param camera_intrinsics: 3x3 ndarray of camera intrinsics
        :param distortion_coefficients: 1x5 ndarray of distortion coeffs.
        :param scale: if you want to resize the image, specify scale factors
        :param rms: max root mean square error when finding grid points
        :param gaussian_sigma: sigma for Gaussian blurring
        :param threshold_window_size: window size for adaptive thresholding
        :param threshold_offset: offset for adaptive thresholding
        :param min_area: minimum area when filtering by area
        :param max_area: maximum area when filtering by area
        :param dot_detector_params: instance of cv2.SimpleBlobDetector_Params()
        """
        super().\
            __init__(scale=scale,
                     camera_intrinsics=camera_intrinsics,
                     distortion_coefficients=distortion_coefficients
                     )

        if len(list_of_indexes) != 4:
            raise ValueError('list_of_index not of length 4')

        self.model_points = model_points
        self.list_of_indexes = list_of_indexes
        self.model_fiducials = self.model_points[self.list_of_indexes]
        self.rms_tolerance = rms
        self.gaussian_sigma = gaussian_sigma
        self.threshold_window_size = threshold_window_size
        self.threshold_offset = threshold_offset
        self.min_area = min_area
        self.max_area = max_area

        self.dot_detector_params = cv2.SimpleBlobDetector_Params()
        self.dot_detector_params.filterByConvexity = False
        self.dot_detector_params.filterByInertia = True
        self.dot_detector_params.filterByCircularity = True
        self.dot_detector_params.minCircularity = 0.5
        self.dot_detector_params.filterByArea = True
        self.dot_detector_params.minArea = self.min_area
        self.dot_detector_params.maxArea = self.max_area

        if dot_detector_params is not None:
            self.dot_detector_params = dot_detector_params

        self.detector = cv2.SimpleBlobDetector_create(self.dot_detector_params)

    def _internal_get_points(self, image, is_distorted=True):
        """
        Extracts points.

        :param image: numpy 2D grey scale image.
        :param is_distorted: False if the input image has already been \
             undistorted.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """

        # pylint:disable=too-many-locals, invalid-name
        # pylint:disable=too-many-statements, too-many-function-args

        default_return = np.zeros((0, 1)), np.zeros((0, 3)), np.zeros((0, 2))

        smoothed = cv2.GaussianBlur(image,
                                    (self.gaussian_sigma, self.gaussian_sigma),
                                    0)

        thresholded = cv2.adaptiveThreshold(smoothed,
                                            255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY,
                                            self.threshold_window_size,
                                            self.threshold_offset)

        # Single blob detection on the smoothed, thresholded image.
        keypoints = self.detector.detect(thresholded)

        if len(keypoints) <= 4:
            return default_return

        # Extract keypoint coordinates (distorted image space).
        distorted_pts = np.array([p.pt for p in keypoints],
                                 dtype=np.float32)
        keypoint_sizes = np.array([p.size for p in keypoints],
                                  dtype=np.float32)

        # Get points in undistorted space for homography estimation.
        if is_distorted:
            pts_for_homography = cv2.undistortPoints(
                distorted_pts.reshape(-1, 1, 2),
                self.camera_intrinsics,
                self.distortion_coefficients,
                P=self.camera_intrinsics
            ).reshape(-1, 2)
        else:
            pts_for_homography = distorted_pts

        # Sort by size and pick biggest 4 as fiducials.
        number_of_keypoints = len(keypoints)
        sorted_indices = keypoint_sizes.argsort()

        biggest_four = np.zeros((4, 5))
        counter = 0
        for idx in sorted_indices[number_of_keypoints - 4:]:
            biggest_four[counter][0] = pts_for_homography[idx, 0]
            biggest_four[counter][1] = pts_for_homography[idx, 1]
            counter += 1

        LOGGER.debug('Biggest 4 points in undistorted space:%s',
                     str(biggest_four))

        # Label which points are below or to the right of the centroid,
        # and assign a score for ordering.
        centroid = np.mean(biggest_four, axis=0)

        for row_counter in range(4):
            if biggest_four[row_counter][1] > centroid[1]:
                biggest_four[row_counter][2] = 1
            if biggest_four[row_counter][0] > centroid[0]:
                biggest_four[row_counter][3] = 1

        for row_counter in range(4):
            biggest_four[row_counter][4] = \
                biggest_four[row_counter][2] * 2 \
                + biggest_four[row_counter][3]

        # Sort so fiducials are: top-left, top-right, bottom-left,
        # bottom-right.
        sorted_fiducials = biggest_four[biggest_four[:, 4].argsort()]

        # Find homography from undistorted fiducials to reference grid.
        homography, _ = \
            cv2.findHomography(sorted_fiducials[:, 0:2],
                               self.model_fiducials[:, 1:3])

        if homography is None:
            LOGGER.warning("Could not compute homography.")
            return default_return

        # Warp all undistorted points into reference space using
        # perspectiveTransform (instead of warping the whole image).
        warped_pts = cv2.perspectiveTransform(
            pts_for_homography.reshape(-1, 1, 2),
            homography
        ).reshape(-1, 2)

        # Match each warped point to the nearest reference grid point.
        model_xy = self.model_points[:, 1:3].astype(np.float32)
        diff = warped_pts[:, np.newaxis, :] - model_xy[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        best_ids = np.argmin(dist_sq, axis=1)

        indexes = self.model_points[best_ids, 0] \
            .reshape(-1, 1).astype(np.int16)
        object_points = self.model_points[best_ids, 3:6]

        min_dists = np.sqrt(np.min(dist_sq, axis=1))
        rms_error = np.mean(min_dists)

        LOGGER.debug('Matching points to reference, RMS=%s', rms_error)
        if rms_error > self.rms_tolerance:
            LOGGER.warning('Matching points to reference, RMS too high')
            return default_return

        # The image points are the original detected coordinates.
        # These are in the distorted input image space (or undistorted
        # if is_distorted=False), which is what the caller expects.
        img_points = distorted_pts.copy()

        # Remove duplicate ID assignments, keeping only unique matches.
        _, unique_idxs, counts = \
            np.unique(indexes, return_index=True, return_counts=True)

        unique_idxs = unique_idxs[counts == 1]

        indexes = indexes[unique_idxs]
        object_points = object_points[unique_idxs]
        img_points = img_points[unique_idxs]

        return indexes, object_points, img_points


    def get_model_points(self):
        """
        If you look in base class, this is expected to return Dict[int, np.ndarray]
        """
        result_dict = {}
        # pylint: disable=consider-using-enumerate
        for i in range(len(self.model_points)):
            point_id = int(self.model_points[i][0])
            point_coords = self.model_points[i][3:6]
            result_dict[point_id] = point_coords
        return result_dict
