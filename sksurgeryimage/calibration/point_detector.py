# coding=utf-8

"""
Base class for a PointDetector.

e.g. Chessboard corners, SIFT points, Charuco points etc.
"""

import logging
import copy
import numpy as np
import cv2

LOGGER = logging.getLogger(__name__)


def _validate_camera_parameters(camera_intrinsics,
                                distortion_coefficients):
    """
    Validates that the camera intrinsics are not None and the correct shape.
    Throws ValueError if incorrect.

    :param camera_intrinsics: [3x3] matrix
    :param distortion_coefficients: [1xn] matrix
    """
    if camera_intrinsics is None:
        raise ValueError('camera_intrinsics is None')
    if distortion_coefficients is None:
        raise ValueError('distortion_coefficients is None')
    if camera_intrinsics.shape[0] != 3:
        raise ValueError('camera_intrinsics does not have 3 rows')
    if camera_intrinsics.shape[1] != 3:
        raise ValueError('camera_intrinsics does not have 3 columns')
    if len(distortion_coefficients.shape) != 1 and \
            distortion_coefficients.shape[0] != 1:
        raise ValueError('distortion_coefficients does not have 1 row')


class PointDetector:
    """
    Class to detect points in a 2D video image.

    These point detectors are often used to detect points for camera
    calibration. However, it would also be possible for some subclasses
    to utilise camera intrinsics and distortion coefficients in order
    to improve the point detection process itself. It would be up to the
    derived class to decide how to use them, if at all.

    :param scale: tuple (x scale, y scale) to scale up/down the image
    :param camera_intrinsics: [3x3] camera matrix
    :param distortion_coefficients: [1xn] distortion coefficients
    """
    def __init__(self,
                 scale=(1, 1),
                 camera_intrinsics=None,
                 distortion_coefficients=None):

        self.scale = scale
        self.scale_x, self.scale_y = scale
        self.camera_intrinsics = None
        self.distortion_coefficients = None
        if camera_intrinsics is not None and distortion_coefficients is None:
            raise ValueError("camera_intrinsics is not None, "
                             "but distortion_coefficients are None??")
        if distortion_coefficients is not None and camera_intrinsics is None:
            raise ValueError("distortion_coefficients are not None, "
                             "but camera intrinsics are None??")
        if camera_intrinsics is not None and \
                distortion_coefficients is not None:
            self.set_camera_parameters(camera_intrinsics,
                                       distortion_coefficients)

    def set_camera_parameters(self,
                              camera_intrinsics,
                              distortion_coefficients):
        """
        Enables camera parameters to be set dynamically at run-time.
        Calls _validate_camera_parameters().

        :param camera_intrinsics: [3x3] camera matrix
        :param distortion_coefficients: [1xn] distortion coefficients
        """
        _validate_camera_parameters(camera_intrinsics,
                                    distortion_coefficients)
        self.camera_intrinsics = copy.deepcopy(camera_intrinsics)
        self.distortion_coefficients = copy.deepcopy(distortion_coefficients)

    def get_camera_parameters(self):
        """
        Returns a copy of the camera matrix, and distortion coefficients.
        Throws RuntimeError if either are None.

        :return: [3x3], [1xn] matrices
        """
        if self.camera_intrinsics is None or \
                self.distortion_coefficients is None:
            raise RuntimeError("Camera parameters are not set")
        tmp_ci = copy.deepcopy(self.camera_intrinsics)
        tmp_dc = copy.deepcopy(self.distortion_coefficients)
        return tmp_ci, tmp_dc

    def get_points(self, image, is_distorted=True):
        """
        Client's call this method to extract points from an image.

        :param image: numpy 2D RGB/grayscale image.
        :param is_distorted: False if the input image has already been \
             undistorted.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """

        if image is None:
            raise TypeError('image is None')
        if not isinstance(image, np.ndarray):
            raise TypeError('image is not an ndarray')

        grey = image
        if len(image.shape) > 2:
            if image.shape[2] == 3:
                grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                grey = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        is_resized = False
        if self.scale_x != 1 or self.scale_y != 1:
            resized = cv2.resize(grey, None, fx=self.scale_x, fy=self.scale_y,
                                 interpolation=cv2.INTER_LINEAR)
            is_resized = True
        else:
            resized = grey

        ids, object_points, image_points = \
            self._internal_get_points(resized, is_distorted=is_distorted)

        if is_resized:
            image_points[:, 0] /= self.scale_x
            image_points[:, 1] /= self.scale_y

        return ids, object_points, image_points

    def _internal_get_points(self, image, is_distorted=True):
        """
        Derived classes override this one.

        :param image: numpy 2D grey scale image.
        :param is_distorted: False if the input image has already been \
             undistorted.
        :return: ids, object_points, image_points as Nx[1,3,2] ndarrays
        """
        raise NotImplementedError()

    def get_model_points(self):
        """
        Derived classes should override this, to detector returns the
        complete model of 3D points. e.g. for a chessboard this would be
        all the corners in chessboard coordinates (e.g. z=0).

        By design, this can return an ndarray with zero rows, if the
        detector does not support 3D coordinates.

        :return: [Nx3] numpy ndarray representing model points.
        """
        raise NotImplementedError()
