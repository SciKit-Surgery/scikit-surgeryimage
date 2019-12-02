# -*- coding: utf-8 -*-

"""
Dotty Grid implementation of PointDetector.
"""

import copy
import logging
import cv2
from collections import Counter
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
                 list_of_indexes, scale=(1, 1)):
        """
        Constructs a PointDetector that extracts a grid of dots,
        with 4 extra large dots, at known locations.

        :param model_points: numpy ndarray of id, x_pix, y_pix, x_mm, y_mm, z_mm
        :param list_of_indexes: list of specific indexes to use as fiducials
        :param scale: if you want to resize the image, specify scale factors
        """
        super(DottyGridPointDetector, self).__init__(scale=scale)

        self.model_points = model_points
        self.list_of_indexes = list_of_indexes
        self.model_fiducials = self.model_points[self.list_of_indexes]

    def _internal_get_points(self, image):
        """
        Extracts points.

        :param image: numpy 2D grey scale image.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """

        smoothed = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 151, 0)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByConvexity = False
        params.filterByInertia = True
        params.filterByCircularity = True
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 50000

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(thresh)

        if len(keypoints) > 4:
            number_of_points = len(keypoints)
            img_points = np.zeros((number_of_points, 2))
            object_points = np.zeros((number_of_points, 3))
            indexes = np.zeros((number_of_points, 1), dtype=np.int16)
            key_points = np.zeros((number_of_points, 3))
            matched_points = np.zeros((number_of_points, 4))

            # Converting OpenCV keypoints to numpy key_points
            counter = 0
            for key in keypoints:
                key_points[counter][0] = key.size
                key_points[counter][1] = key.pt[0]
                key_points[counter][2] = key.pt[1]
                counter = counter + 1

            # Sort keypoints and pick biggest 4
            sorted_points = key_points[key_points[:, 0].argsort()]

            biggest_four = np.zeros((4, 5))
            counter = 0
            for row_counter in range(number_of_points - 4, number_of_points):
                biggest_four[counter][0] = sorted_points[row_counter][1]
                biggest_four[counter][1] = sorted_points[row_counter][2]
                counter = counter + 1

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

            # Transform the detected points to the standard grid.
            homography, status = \
                cv2.findHomography(sorted_fiducials[:, 0:2],
                                   self.model_fiducials[:, 1:3])

            float_array = key_points[:, 1:3] \
                .astype(np.float32) \
                .reshape(-1, 1, 2)

            transformed_points = \
                cv2.perspectiveTransform(float_array,
                                         homography)

            # For each transformed point, find closest point in standard grid.
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
                img_points[counter][0] = key_points[counter][1]
                img_points[counter][1] = key_points[counter][2]
                object_points[counter][0] = self.model_points[best_id_so_far][3]
                object_points[counter][1] = self.model_points[best_id_so_far][4]
                object_points[counter][2] = self.model_points[best_id_so_far][5]
                matched_points[counter][0] = key_points[counter][1]
                matched_points[counter][1] = key_points[counter][2]
                matched_points[counter][2] = self.model_points[best_id_so_far][1]
                matched_points[counter][3] = self.model_points[best_id_so_far][2]
                rms_error = rms_error + best_distance_so_far
                counter = counter + 1

            # Now recompute homography using all points so far.
            homography, status = \
                cv2.findHomography(matched_points[:, 0:2],
                                   matched_points[:, 2:4])

            # And re-transform points using new homography
            transformed_points = \
                cv2.perspectiveTransform(float_array,
                                         homography)

            reference_points = copy.deepcopy(transformed_points)

            # Now re-find matching points
            rms_error = 0
            counter = 0
            indexes_as_list = []
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
                        reference_points[counter][0][0] = self.model_points[best_id_so_far][1]
                        reference_points[counter][0][1] = self.model_points[best_id_so_far][2]

                indexes_as_list.append(best_id_so_far)
                indexes[counter] = self.model_points[best_id_so_far][0]
                img_points[counter][0] = key_points[counter][1]
                img_points[counter][1] = key_points[counter][2]
                object_points[counter][0] = self.model_points[best_id_so_far][3]
                object_points[counter][1] = self.model_points[best_id_so_far][4]
                object_points[counter][2] = self.model_points[best_id_so_far][5]
                matched_points[counter][0] = key_points[counter][1]
                matched_points[counter][1] = key_points[counter][2]
                matched_points[counter][2] = self.model_points[best_id_so_far][1]
                matched_points[counter][3] = self.model_points[best_id_so_far][2]
                rms_error = rms_error + best_distance_so_far
                counter = counter + 1

            # invert matched points
            inverse_homography = np.linalg.inv(homography)
            inverted_points = \
                cv2.perspectiveTransform(reference_points,
                                         inverse_homography)

            # Print original and inverted
            for counter in range(number_of_points):
                dist = (key_points[counter][1] - inverted_points[counter][0][0]) \
                       * (key_points[counter][1] - inverted_points[counter][0][0]) \
                       + (key_points[counter][2] - inverted_points[counter][0][1]) \
                       * (key_points[counter][2] - inverted_points[counter][0][1])
#                print("Matt, c=" + str(counter)
#                      + ', i=' + str(indexes[counter])
#                      + ', o=' + str(key_points[counter][1]) + ', ' + str(key_points[counter][2])
#                      + ', r=' + str(inverted_points[counter][0][0]) + ', ' + str(inverted_points[counter][0][1])
#                      + ', d=' + str(dist)
#                      )

            # Find duplicates
            counted_indexes = Counter(indexes_as_list)

            # Work out which indexes are suspicious
            dodgy_indexes = []
            for c_i in counted_indexes:
                if counted_indexes[c_i] > 1:
                    dodgy_indexes.append(c_i)

            print("Matt, duplicates=" + str(dodgy_indexes))

            # Work out which rows to delete from output
            dodgy_rows = []
            for d_i in dodgy_indexes:
                # Find best case for this index
                best_distance_so_far = np.finfo('d').max
                best_row_index_so_far = -1
                for counter in range(number_of_points):
                    if indexes[counter] == d_i:
                        dist = (key_points[counter][1] - inverted_points[counter][0][0])\
                            * (key_points[counter][1] - inverted_points[counter][0][0])\
                            + (key_points[counter][2] - inverted_points[counter][0][1])\
                            * (key_points[counter][2] - inverted_points[counter][0][1])
                        if dist < best_distance_so_far:
                            best_distance_so_far = dist
                            best_row_index_so_far = counter
                # Find all other instances of this index
                for counter in range(number_of_points):
                    if indexes[counter] == d_i and counter != best_row_index_so_far:
                        dodgy_rows.append(counter)

            # Delete dodgy rows
            indexes = np.delete(indexes, dodgy_rows, axis=0)
            object_points = np.delete(object_points, dodgy_rows, axis=0)
            img_points = np.delete(img_points, dodgy_rows, axis=0)

            # Compute total RMS error, to see if fit was good enough.
            rms_error = rms_error / number_of_points
            rms_error = np.sqrt(rms_error)

            print("Matt, rms=" + str(rms_error))

            if rms_error < 25:
                return indexes, object_points, img_points

        # If we didn't find all points, of the fit was poor,
        # return a consistent set of 'nothing'
        return np.zeros((0, 1)), np.zeros((0, 3)), np.zeros((0, 2))
