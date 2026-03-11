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
                 reference_image_size=None,
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
        :param reference_image_size: used to warp undistorted image to reference
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
        if reference_image_size is None:
            raise ValueError('You must provide a reference image size')

        self.model_points = model_points
        self.list_of_indexes = list_of_indexes
        self.model_fiducials = self.model_points[self.list_of_indexes]
        self.reference_image_size = reference_image_size
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

    def _internal_get_points(self, image, is_distorted=True):
        """
        Extracts points.

        :param image: numpy 2D grey scale image.
        :param is_distorted: False if the input image has already been \
             undistorted.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """

        # pylint:disable=too-many-locals, invalid-name, too-many-branches
        # pylint:disable=too-many-statements

        # If we didn't find all points, of the fit was poor,
        # return a consistent set of 'nothing'
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

        # Detect points in the distorted image
        detector = cv2.SimpleBlobDetector_create(self.dot_detector_params)
        keypoints = detector.detect(thresholded)

        # If input image is distorted, undistort and also detect points
        # in undistorted image.
        if is_distorted:
            undistorted_image = cv2.undistort(smoothed,
                                              self.camera_intrinsics,
                                              self.distortion_coefficients
                                              )

            undistorted_thresholded = \
                cv2.adaptiveThreshold(undistorted_image,
                                      255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY,
                                      self.threshold_window_size,
                                      self.threshold_offset)

            undistorted_keypoints = detector.detect(undistorted_thresholded)

        else:
            undistorted_image = image
            undistorted_keypoints = keypoints

        # Note that keypoints and undistorted_keypoints
        # can be of different length
        if len(keypoints) <= 4 or len(undistorted_keypoints) <= 4:
            return default_return

        number_of_undistorted_keypoints = len(undistorted_keypoints)
        undistorted_key_points = np.array([(p.size, p.pt[0], p.pt[1]) for p in undistorted_keypoints], dtype=np.float32)

        # Sort undistorted_key_points and pick biggest 4
        sorted_points = undistorted_key_points[
            undistorted_key_points[:, 0].argsort()]

        biggest_four = np.zeros((4, 5))
        counter = 0
        for row_counter in range(number_of_undistorted_keypoints - 4,
                                 number_of_undistorted_keypoints):
            biggest_four[counter][0] = sorted_points[row_counter][1]
            biggest_four[counter][1] = sorted_points[row_counter][2]
            counter = counter + 1

        LOGGER.debug('Biggest 4 points in undistorted image:%s',
                     str(biggest_four))

        # Labelling which points are below or to the right of the centroid,
        # and assigning a score.
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

        # Then we sort by this score, so the fiducials are
        # top left, top right, bottom left, bottom right.
        sorted_fiducials = biggest_four[biggest_four[:, 4].argsort()]

        # Find the homography between the distortion corrected points
        # and the reference points, from an ideal face-on image.
        homography, _ = \
            cv2.findHomography(sorted_fiducials[:, 0:2],
                               self.model_fiducials[:, 1:3])

        # Warp image to cannonical face on.
        warped = cv2.warpPerspective(undistorted_image,
                                     homography,
                                     self.reference_image_size)
        warped_keypoints = detector.detect(warped)
        number_of_warped_keypoints = len(warped_keypoints)
        warped_key_points = np.array([(p.size, p.pt[0], p.pt[1])
                                      for p in warped_keypoints], dtype=np.float32)
        img_points = np.zeros((number_of_warped_keypoints, 2))

        # Note, warped_key_points and undistorted_key_points
        # have different order.

        float_array = warped_key_points[:, 1:3] \
            .astype(np.float32) \
            .reshape(-1, 1, 2)

        transformed_points = \
            cv2.perspectiveTransform(float_array,
                                     np.eye(3))

        if transformed_points is None:
            LOGGER.info("transformed_points is None, skipping")
            return default_return

        inverted_points = \
            cv2.perspectiveTransform(transformed_points,
                                     np.linalg.inv(homography))

        # For each transformed point, find closest point in reference grid.
        warped_pts = np.array([p.pt for p in warped_keypoints], dtype=np.float32)
        model_xy = self.model_points[:, 1:3].astype(np.float32)
        diff = warped_pts[:, np.newaxis, :] - model_xy[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        best_ids = np.argmin(dist_sq, axis=1)
        indexes = self.model_points[best_ids, 0].reshape(-1, 1).astype(np.int16)
        object_points = self.model_points[best_ids, 3:6]
        matched_points = np.zeros((len(warped_pts), 4), dtype=np.float32)
        matched_points[:, 0:2] = warped_pts
        matched_points[:, 2:4] = self.model_points[best_ids, 1:3]
        min_dists = np.sqrt(np.min(dist_sq, axis=1))
        rms_error = np.mean(min_dists)
        LOGGER.debug('Matching points to reference, RMS=%s', rms_error)

        if rms_error > self.rms_tolerance:
            LOGGER.warning('Matching points to reference, RMS too high')
            return default_return

        # Now copy inverted points into matched_points
        flattened_inverted = inverted_points.reshape(-1, 2)
        matched_points[:, 0:2] = flattened_inverted
        img_points[:, 0:2] = flattened_inverted

        if is_distorted:
            # Input image was a distorted image, so now we have to map
            # undistorted points back to distorted points.
            fx = self.camera_intrinsics[0, 0]
            fy = self.camera_intrinsics[1, 1]
            cx = self.camera_intrinsics[0, 2]
            cy = self.camera_intrinsics[1, 2]

            rel_x = (matched_points[:, 0] - cx) / fx
            rel_y = (matched_points[:, 1] - cy) / fy
            r2 = rel_x**2 + rel_y**2
            r4 = r2 ** 2
            r6 = r2 * r4
            k1, k2, p1, p2, k3 = self.distortion_coefficients

            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

            dist_x = rel_x * radial + (2 * p1 * rel_x * rel_y + p2 * (r2 + 2 * rel_x**2))
            dist_y = rel_y * radial + (p1 * (r2 + 2 * rel_y**2) + 2 * p2 * rel_x * rel_y)
            img_points = np.zeros((len(matched_points), 2), dtype=np.float32)
            img_points[:, 0] = dist_x * fx + cx
            img_points[:, 1] = dist_y * fy + cy

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
