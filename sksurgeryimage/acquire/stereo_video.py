# coding=utf-8

"""
Module for stereo video source acquisition.
"""

import logging
import cv2
import sksurgerycore.utilities.validate as scv
import sksurgerycore.utilities.validate_matrix as scvm
import sksurgeryimage.acquire.video_source as vs
import sksurgeryimage.processing.interlace as i


LOGGER = logging.getLogger(__name__)


class StereoVideoLayouts:
    """
    Class to hold some constants, like an enum.
    """
    DUAL = 0
    INTERLACED = 1
    VERTICAL = 2


class StereoVideo:
    """
    Provides a convenient object to manage various stereo input styles.
    Developed firstly for laparoscopic surgery, but broadly applicable
    to any stereo setup using our TimestampedVideoSource and VideoSourceWrapper.

    Design Principles:

    1. Fail early, throwing exceptions for all errors.
    2. Works with or without camera parameters.
    3. If no camera parameters, calling get_undistorted()
       or get_rectified() is an Error.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, layout, channels, dims=None):
        """
        Constructor, for stereo video sources of the same size.

        Originally designed for

        - Storz 3D laparoscope: two separate channels interlaced
          at 1920x1080, and fed into channel_1.
        - Viking 3D laparoscope: separate left and right channels,
          both at 1920x1080, fed into into AJA Hi5-3D,
          stacking channels top and bottom each at 1920x540,
          resulting in 1920x1080, and fed into channel_1.
        - Viking 3D laparoscope: two separate 1920x1080 channels
          fed into channel_1 and channel_2.
        - DaVinci laparoscope: two separate channels (resolution?)
          fed into channel_1 and channel_2.


        :param layout: See StereoVideoLayouts.
        :param channels: list of camera integer id's, or string file path name
        :param dims: (width, height) - required size in pixels
        """
        if layout is not StereoVideoLayouts.DUAL \
           and layout is not StereoVideoLayouts.INTERLACED \
           and layout is not StereoVideoLayouts.VERTICAL:
            raise ValueError("Layout must be either StereoVideoLayouts.DUAL, "
                             + "StereoVideoLayouts.INTERLACED "
                             + " or StereoVideoLayoutsVERTICAL.")

        if not channels:
            raise ValueError("You must provide at least one channel of input.")

        if len(channels) != 1 and len(channels) != 2:
            raise ValueError("You must provide either 1 or "
                             + "2 channels of input.")

        if len(channels) >= 1:
            if channels[0] is None:
                raise ValueError("First channel is None.")
            scv.validate_is_string_or_number(channels[0])

        if len(channels) == 2:
            if channels[1] is None:
                raise ValueError("Second channel is None.")
            scv.validate_is_string_or_number(channels[1])

        if layout == StereoVideoLayouts.DUAL and len(channels) != 2:
            raise ValueError("If you specify layout to be DUAL, "
                             + "you must provide 2 channels.")

        # Further validation of (width, height)
        if dims is not None:
            scv.validate_width_height(dims)

        self.layout = layout
        self.channels = channels
        self.scaling = [1, 2]
        self.camera_matrices = [None, None]
        self.distortion_coefficients = [None, None]
        self.stereo_rotation = None
        self.stereo_translation = None
        self.rectify_rotation = [None, None]
        self.rectify_projection = [None, None]
        self.rectify_q = None
        self.rectify_new_size = None
        self.rectify_valid_roi = [None, None]
        self.rectify_dx = [None, None]
        self.rectify_dy = [None, None]
        self.rectify_initialised = False

        self.video_sources = vs.VideoSourceWrapper()
        self.video_sources.add_source(channels[0], dims)
        if len(channels) == 2:
            self.video_sources.add_source(channels[1], dims)
            self.scaling = [1, 1]

    def set_intrinsic_parameters(self,
                                 camera_matrices,
                                 distortion_coefficients):
        """
        Sets both sets of intrinsic parameters.

        :param camera_matrices: list of 2, 3x3 numpy arrays.
        :param distortion_coefficients: list of 2, 1xN numpy arrays.
        :raises: ValueError, TypeError
        """
        if len(camera_matrices) != 2:
            raise ValueError("There should be exactly 2 camera matrices.")
        if len(distortion_coefficients) != 2:
            raise ValueError("There should be exactly 2 "
                             + "sets of distortion coefficients.")

        # Further validation of camera matrices and distortion coefficients.
        for matrix in camera_matrices:
            scvm.validate_camera_matrix(matrix)
        for coefficients in distortion_coefficients:
            scvm.validate_distortion_coefficients(coefficients)

        self.camera_matrices = camera_matrices
        self.distortion_coefficients = distortion_coefficients
        self.rectify_initialised = False

    def set_extrinsic_parameters(self,
                                 rotation,
                                 translation,
                                 dims
                                 ):
        """
        Sets the stereo extrinsic parameters.

        :param rotation: 3x3 numpy array representing rotation matrix.
        :param translation: 3x1 numpy array representing translation vector.
        :param dims: new image size for rectification
        :raises: ValueError, TypeError
        """
        scvm.validate_rotation_matrix(rotation)
        scvm.validate_translation_column_vector(translation)
        scv.validate_width_height(dims)

        self.stereo_rotation = rotation
        self.stereo_translation = translation
        self.rectify_new_size = dims
        self.rectify_initialised = False

    def release(self):
        """
        Asks internal VideoSourceWrapper to release all sources.
        """
        self.video_sources.release_all_sources()

    def grab(self):
        """
        Asks internal VideoSourceWrapper to grab images.
        """
        self.video_sources.grab()

    def retrieve(self):
        """
        Asks internal VideoSourceWrapper to retrieve images.
        """
        self.video_sources.retrieve()

    def get_images(self):
        """
        Returns the 2 channels, unscaled, as a list of images.

        :return: list of images
        """
        return self._extract_separate_views()

    def get_scaled(self):
        """
        Returns the 2 channels, scaled, as a list of images.

        :return: list of images
        """
        frames = self.get_images()
        scaled = []
        if len(self.channels) == 1:  # stereo frames provided in one image
            for frame in frames:
                scaled_image = cv2.resize(frame,
                                          None,
                                          fx=self.scaling[0],
                                          fy=self.scaling[1],
                                          interpolation=cv2.INTER_NEAREST)
                scaled.append(scaled_image)
        else:
            scaled = frames
        return scaled

    def get_undistorted(self):
        """
        Returns the 2 channels, undistorted, as a list of images.

        :return: list of images
        :raises: ValueError - if you haven't already provided camera parameters
        """
        self._validate_intrinsic_params()
        frames = self.get_scaled()
        undistorted = []
        counter = 0
        for frame in frames:
            undist = cv2.undistort(frame,
                                   self.camera_matrices[counter],
                                   self.distortion_coefficients[counter]
                                   )
            undistorted.append(undist)
            counter += 1
        return undistorted

    def get_rectified(self):
        """
        Returns the 2 channels, rectified, as a list of images.

        :return: list of images
        :raises: ValueError, TypeError - if camera parameters are not set.
        """
        self._validate_intrinsic_params()
        scvm.validate_rotation_matrix(self.stereo_rotation)
        scvm.validate_translation_column_vector(self.stereo_translation)

        frames = self.get_scaled()

        if not self.rectify_initialised:

            image_size = (frames[0].shape[1], frames[0].shape[0])

            self.rectify_rotation[0], \
                self.rectify_rotation[1], \
                self.rectify_projection[0], \
                self.rectify_projection[1], \
                self.rectify_q, \
                self.rectify_valid_roi[0], \
                self.rectify_valid_roi[1] = \
                cv2.stereoRectify(self.camera_matrices[0],
                                  self.distortion_coefficients[0],
                                  self.camera_matrices[1],
                                  self.distortion_coefficients[1],
                                  image_size,
                                  self.stereo_rotation,
                                  self.stereo_translation,
                                  flags=cv2.CALIB_ZERO_DISPARITY,
                                  alpha=0,
                                  newImageSize=self.rectify_new_size
                                  )
            for image_index in [0, 1]:
                self.rectify_dx[image_index], self.rectify_dy[image_index] = \
                    cv2.initUndistortRectifyMap(
                        self.camera_matrices[image_index],
                        self.distortion_coefficients[image_index],
                        self.rectify_rotation[image_index],
                        self.rectify_projection[image_index],
                        self.rectify_new_size,
                        cv2.CV_32FC1
                        )

            self.rectify_initialised = True

        rectified = []
        counter = 0
        for frame in frames:
            rectified_image = cv2.remap(frame,
                                        self.rectify_dx[counter],
                                        self.rectify_dy[counter],
                                        cv2.INTER_LINEAR
                                        )
            rectified.append(rectified_image)
            counter += 1
        return rectified

    def _validate_intrinsic_params(self):
        """
        Internal method to ensure we have camera parameters.

        :raises ValueError: if you haven't already provided camera parameters.
        """
        if self.camera_matrices[0] is None \
           or self.camera_matrices[1] is None \
           or self.distortion_coefficients[0] is None \
           or self.distortion_coefficients[1] is None:
            raise ValueError("Not all camera parameters are available")
        return True

    def _extract_separate_views(self):
        """
        Internal method to separate vertically stacked or interlaced frames.

        :return: either [top, bottom], or [even, odd] images
        """
        if not self.video_sources.frames:
            raise RuntimeError("No frames present, did you "
                               + "call grab and retrieve yet?")

        if len(self.video_sources.frames) > 1:
            return self.video_sources.frames

        if self.layout == StereoVideoLayouts.INTERLACED:
            even_rows, odd_rows \
                = i.deinterlace_to_view(self.video_sources.frames[0])
            separated = [even_rows, odd_rows]
            return separated

        top, bottom \
            = i.split_stacked_to_view(self.video_sources.frames[0])
        separated = [top, bottom]
        return separated
