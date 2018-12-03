# coding=utf-8

"""
Module for stereo video source acquisition.
"""

import logging
import cv2
import numpy as np
import sksurgeryimage.acquire.video_source as vs
import sksurgeryimage.processing.interlace as i
import sksurgeryimage.utilities.utilities as u

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
    to any stereo setup using our VideoSource and VideoSourceWrapper.

    Design Principles:

        1. Fail early, throwing exceptions for all errors.
        2. Works with or without camera parameters.
        3. If no camera parameters, calling get_undistorted() or get_rectified() is an Error.
    """

    def __init__(self, layout, channels, dims=None):
        """
        Constructor, for stereo video sources of the same size.

        Originally designed for

            Storz 3D laparoscope: two separate channels interlaced at 1920x1080, and fed into channel_1.
            Viking 3D laparoscope: separate left and right channels, both at 1920x1080, fed into into AJA Hi5-3D,
                stacking channels top and bottom each at 1920x540, resulting in 1920x1080, and fed into channel_1.
            Viking 3D laparoscope: two separate 1920x1080 channels fed into channel_1 and channel_2.
            DaVinci laparoscope: two separate channels (resolution?) fed into channel_1 and channel_2.


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
            raise ValueError("You must provide either 1 or 2 channels of input.")

        if len(channels) >= 1:
            if channels[0] is None:
                raise ValueError("First channel is None.")
            if not u.is_string_or_number(channels[0]):
                raise TypeError("First channel descriptor is not a file path or camera index.")

        if len(channels) == 2:
            if channels[1] is None:
                raise ValueError("Second channel is None.")
            if not u.is_string_or_number(channels[1]):
                raise TypeError("Second channel descriptor is not a file path or camera index.")

        if layout == StereoVideoLayouts.DUAL and len(channels) != 2:
            raise ValueError("If you specify layout to be DUAL, you must provide 2 channels.")

        # Further validation of (width, height)
        if dims is not None:
            u.validate_width_height(dims)

        self.layout = layout
        self.channels = channels
        self.camera_matrices = [None, None]
        self.distortion_coefficients = [None, None]

        self.video_sources = vs.VideoSourceWrapper()

        self.video_sources.add_source(channels[0], dims)
        if len(channels) == 2:
            self.video_sources.add_source(channels[1], dims)

    def set_camera_parameters(self,
                              camera_matrices,
                              distortion_coefficients):
        """
        Sets stereo camera parameters.

        :param camera_matrices: list of 2, 3x3 numpy arrays.
        :param distortion_coefficients: list of 3, 1xN numpy arrays.
        :return:
        """
        if len(camera_matrices) != 2:
            raise ValueError("There should be exactly 2 camera matrices.")
        if len(distortion_coefficients) != 2:
            raise ValueError("There should be exactly 2 "
                             + "sets of distortion coefficients.")
        for c in camera_matrices:
            u.validate_camera_matrix(c)
        for d in distortion_coefficients:
            u.validate_distortion_coefficients(d)

        self.camera_matrices = camera_matrices
        self.distortion_coefficients = distortion_coefficients

    def release(self):
        """
        Asks internal VideoSourceWrapper to release all sources.
        """
        self.video_sources.release_all_sources()

    def grab(self):
        """
        Asks internal VideoSourceWrapper to grab images. This doesn't
        actually do any decoding. You are expected to call get(),
        get_undistorted(), get_rectified() next, depending on your use-case.
        """
        self.video_sources.grab()

    def get(self):
        """
        Returns the 2 channels, scaled, as a list of images.

        :return: list of images
        """
        frames = self._extract_separate_views()
        scaled = []
        if len(channels) == 1:  # stereo frames provided in one image
            for f in frames:
                s = cv2.resize(f, None, fx=1, fy=2, interpolation=cv2.INTER_LINEAR)
                scaled.append(s)
        else:
            scaled = frames
        return scaled

    def get_undistorted(self):
        """
        Returns the 2 channels, undistorted, as a list of images.

        :return: list of images
        :raises ValueError: if you haven't already provided camera parameters
        """
        self._validate_camera_params()
        frames = self.get_views()
        undistorted = []
        counter = 0
        for f in frames:
            u = cv2.undistort(f,
                              self.camera_matrices[counter],
                              self.distortion_coefficients[counter]
                              )
            undistorted.append(u)
            counter += 1
        return undistorted

    def get_rectified(self):
        """
        Returns the 2 channels, rectified, as a list of images.

        :return: list of images
        :raises ValueError: if you haven't already provided camera parameters.
        """
        self._validate_camera_params()
        frames = self._extract_separate_views()
        rectified = []
        counter = 0
        for f in frames:
            r = cv2.rectify(f)
            rectified.append(r)
            counter += 1
        return rectified

    def _validate_camera_params(self):
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
        if len(self.video_sources.frames) > 1:
            return self.video_sources.frames
        else:
            if self.layout == StereoVideoLayouts.INTERLACED:
                even_rows, odd_rows \
                    = i.deinterlace_to_view(self.video_sources.frames[0])
                separated = [even_rows, odd_rows]
                return separated
            else:
                top, bottom \
                    = i.split_stacked_to_view(self.video_sources.frames[0])
                separated = [top, bottom]
                return separated
