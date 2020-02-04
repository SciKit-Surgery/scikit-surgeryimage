# -*- coding: utf-8 -*-

"""
Dotty Grid implementation of PointDetector.
"""

import logging
import cv2
import numpy as np
from sksurgeryimage.processing.point_detector import PointDetector

LOGGER = logging.getLogger(__name__)


class DottyGridPointDetector(PointDetector):
    """
    Class to detect a grid of dots in a 2D grey scale video image.

    More specifically, a grid of dots with 4 larger dots at known locations.
    """
    def __init__(self,
                 model_points,
                 list_of_indexes,
                 intrinsics,
                 distortion_coefficients,
                 scale=(1, 1),
                 reference_image_size=None,
                 rms=30
                 ):
        """
        Constructs a PointDetector that extracts a grid of dots,
        with 4 extra large dots, at known locations.

        Requires intrinsics and distortion_coefficients to be provided,
        then these are used as a reference transform to undistort
        the image, which makes matching to a reference grid more reliable.

        The list of indexes, must be of length 4, and correspond to
        top-left, top-right, bottom-left, bottom-right bigger blobs.

        :param model_points: numpy ndarray of id, x_pix, y_pix, x_mm, y_mm, z_mm
        :param list_of_indexes: list of specific indexes to use as fiducials
        :param intrinsics: 3x3 ndarray of camera intrinsics
        :param distortion_coefficients: 1x5 ndarray of distortion coeffs.
        :param scale: if you want to resize the image, specify scale factors
        :param reference_image_size: used to warp undistorted image to reference
        :param rms: max root mean square error when finding grid points
        """
        super(DottyGridPointDetector, self).__init__(scale=scale)

        if intrinsics is None:
            raise ValueError('intrinsics is None')
        if distortion_coefficients is None:
            raise ValueError('distortion coefficients are None')
        if len(distortion_coefficients) != 5:
            raise ValueError('there should be 5 distortion coefficients')
        if len(list_of_indexes) != 4:
            raise ValueError('list_of_index not of length 4')
        if reference_image_size is None:
            raise ValueError('You must provide a reference image size')

        self.model_points = model_points
        self.list_of_indexes = list_of_indexes
        self.intrinsics = intrinsics
        self.distortion_coefficients = distortion_coefficients
        self.model_fiducials = self.model_points[self.list_of_indexes]
        self.reference_image_size = reference_image_size
        self.rms_tolerance = rms

    def _internal_get_points(self, image):
        """
        Extracts points.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """

        # pylint:disable=too-many-locals, invalid-name, too-many-branches
        # pylint:disable=too-many-statements

        smoothed = cv2.GaussianBlur(image, (5, 5), 0)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByConvexity = False
        params.filterByInertia = True
        params.filterByCircularity = True
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 50000

        # Detect points in the distorted image
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(smoothed)

        # Also detect them in an undistorted image.
        undistorted_image = cv2.undistort(smoothed,
                                          self.intrinsics,
                                          self.distortion_coefficients
                                          )
        undistorted_keypoints = detector.detect(undistorted_image)

        # Note that keypoints and undistorted_keypoints
        # can be of different length
        if len(keypoints) > 4 and len(undistorted_keypoints) > 4:
            number_of_keypoints = len(keypoints)
            number_of_undistorted_keypoints = len(undistorted_keypoints)

            # These are for intermediate storage.
            key_points = \
                np.zeros((number_of_keypoints, 3))
            undistorted_key_points = \
                np.zeros((number_of_undistorted_keypoints, 3))

            # Converting OpenCV keypoints to numpy key_points
            counter = 0
            for key in keypoints:
                key_points[counter][0] = key.size
                key_points[counter][1] = key.pt[0]
                key_points[counter][2] = key.pt[1]
                counter = counter + 1
            counter = 0
            for key in undistorted_keypoints:
                undistorted_key_points[counter][0] = key.size
                undistorted_key_points[counter][1] = key.pt[0]
                undistorted_key_points[counter][2] = key.pt[1]
                counter = counter + 1

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
            warped_key_points = \
                np.zeros((number_of_warped_keypoints, 3))
            counter = 0
            for key in warped_keypoints:
                warped_key_points[counter][0] = key.size
                warped_key_points[counter][1] = key.pt[0]
                warped_key_points[counter][2] = key.pt[1]
                counter = counter + 1
            img_points = np.zeros((number_of_warped_keypoints, 2))
            object_points = np.zeros((number_of_warped_keypoints, 3))
            indexes = np.zeros((number_of_warped_keypoints, 1),
                               dtype=np.int16)
            matched_points = \
                np.zeros((number_of_warped_keypoints, 4))

            # Note, warped_key_points and undistorted_key_points
            # have different order.

            float_array = warped_key_points[:, 1:3] \
                .astype(np.float32) \
                .reshape(-1, 1, 2)

            transformed_points = \
                cv2.perspectiveTransform(float_array,
                                         np.eye(3))

            inverted_points = \
                cv2.perspectiveTransform(transformed_points,
                                         np.linalg.inv(homography))

            # For each transformed point, find closest point in reference grid.
            rms_error = 0
            counter = 0
            for transformed_point in transformed_points:
                best_distance_so_far = np.finfo('d').max
                best_id_so_far = -1
                for model_point_counter in range(self.model_points.shape[0]):
                    sq_dist = (self.model_points[model_point_counter][1]
                               - transformed_point[0][0]) \
                            * (self.model_points[model_point_counter][1]
                               - transformed_point[0][0]) \
                            + (self.model_points[model_point_counter][2]
                               - transformed_point[0][1]) \
                            * (self.model_points[model_point_counter][2]
                               - transformed_point[0][1])

                    if sq_dist < best_distance_so_far:
                        best_id_so_far = model_point_counter
                        best_distance_so_far = sq_dist

                indexes[counter] = self.model_points[best_id_so_far][0]
                object_points[counter][0] = self.model_points[best_id_so_far][3]
                object_points[counter][1] = self.model_points[best_id_so_far][4]
                object_points[counter][2] = self.model_points[best_id_so_far][5]
                matched_points[counter][0] = \
                    warped_key_points[counter][1]
                matched_points[counter][1] = \
                    warped_key_points[counter][2]
                matched_points[counter][2] = \
                    self.model_points[best_id_so_far][1]
                matched_points[counter][3] = \
                    self.model_points[best_id_so_far][2]
                rms_error = rms_error + best_distance_so_far
                counter = counter + 1

            # Compute total RMS error, to see if fit was good enough.
            rms_error = rms_error / number_of_undistorted_keypoints
            rms_error = np.sqrt(rms_error)

            LOGGER.debug('Matching points to reference, RMS=%s', rms_error)

            if rms_error > self.rms_tolerance:
                LOGGER.warning('Matching points to reference, RMS too high')
                return np.zeros((0, 1)), np.zeros((0, 3)), np.zeros((0, 2))

            # Now copy inverted points into matched_points
            # pylint: disable=consider-using-enumerate
            for counter in range(len(inverted_points)):
                matched_points[counter][0] = inverted_points[counter][0][0]
                matched_points[counter][1] = inverted_points[counter][0][1]

            # Now have to map undistorted points back to distorted points
            for counter in range(number_of_warped_keypoints):

                # Distort point to match original input image.
                relative_x = (matched_points[counter][0]
                              - self.intrinsics[0][2]) / self.intrinsics[0][0]
                relative_y = (matched_points[counter][1]
                              - self.intrinsics[1][2]) / self.intrinsics[1][1]
                r2 = relative_x * relative_x + relative_y * relative_y
                radial = (1
                          + self.distortion_coefficients[0] * r2
                          + self.distortion_coefficients[1] * r2 * r2
                          + self.distortion_coefficients[4] * r2 * r2 * r2
                          )
                distorted_x = relative_x * radial
                distorted_y = relative_y * radial

                distorted_x = distorted_x + (2 * self.distortion_coefficients[2]
                                             * relative_x * relative_y
                                             + self.distortion_coefficients[3]
                                             * (r2 + 2
                                                * relative_x
                                                * relative_x))

                distorted_y = distorted_y + (self.distortion_coefficients[2]
                                             * (r2 + 2 * relative_y
                                                * relative_y)
                                             + 2 *
                                             self.distortion_coefficients[3]
                                             * relative_x * relative_y)

                distorted_x = distorted_x * self.intrinsics[0][0] \
                    + self.intrinsics[0][2]
                distorted_y = distorted_y * self.intrinsics[1][1] \
                    + self.intrinsics[1][2]

                img_points[counter][0] = distorted_x
                img_points[counter][1] = distorted_y

            return indexes, object_points, img_points

        # If we didn't find all points, of the fit was poor,
        # return a consistent set of 'nothing'
        return np.zeros((0, 1)), np.zeros((0, 3)), np.zeros((0, 2))
