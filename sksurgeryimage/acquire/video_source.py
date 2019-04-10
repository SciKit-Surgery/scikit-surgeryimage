# coding=utf-8
#pylint:disable=no-name-in-module

"""
Module for video source acquisition.
Classes capture data from a video source into a numpy array.
"""

import logging
import datetime
import cv2
import numpy as np
import sksurgerycore.utilities.validate_file as vf
import sksurgeryimage.utilities.camera_utilities as cu

LOGGER = logging.getLogger(__name__)

class TimestampedVideoSource:
    """
    Capture and store data from camera/file source.
    Augments the cv2.VideoCapture() to provide passing of
    camera dimensions in constructor, and storage of frame data.
    """
    def __init__(self, source_num_or_file, dims=None):
        """
        Constructs a TimestampedVideoSource.

        :param source_num_or_file: integer camera number or file path
        :param dims: optional (width, height) as a pair of integers
        """
        self.source = cv2.VideoCapture(source_num_or_file)
        self.timestamp = None

        if not self.source.isOpened():
            raise RuntimeError("Failed to open Video camera:"
                               + str(source_num_or_file))

        self.source_name = source_num_or_file

        LOGGER.info("Adding input from source: %s", self.source_name)

        if dims:

            width, height = dims

            if not isinstance(width, int):
                raise TypeError("Width must be an integer")
            if not isinstance(height, int):
                raise TypeError("Height must be an integer")
            if width < 1:
                raise ValueError("Width must be >= 1")
            if height < 1:
                raise ValueError("Height must be >= 1")

            self.source.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.source.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        else:
            width = int(self.source.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.source.get(cv2.CAP_PROP_FRAME_HEIGHT))

        LOGGER.info("Source dimensions %s %s", width, height)

        self.frame = np.empty((height, width, 3), dtype=np.uint8)
        self.ret = None

    def grab(self):
        """
        Call the cv2.VideoCapture grab function and get a timestamp.
        """
        self.ret = self.source.grab()

        self.timestamp = datetime.datetime.now()

        return self.ret

    def retrieve(self):
        """
        Call the cv2.VideoCapture retrieve function and
        store the returned frame.
        """
        self.ret, self.frame = self.source.retrieve()
        return self.ret, self.frame

    def read(self):
        """
        Do a grab(), then retrieve() operation.
        """
        self.grab()
        self.retrieve()

        return self.ret, self.frame

    def isOpened(self):
        """ Call the cv2.VideoCapture isOpened function.
        """
        # pylint: disable=invalid-name
        # using isOpened to be consistent with OpenCV function name
        return self.source.isOpened()

    def release(self):
        """
        Release the cv2.VideoCapture source.
        """
        self.source.release()

class VideoSourceWrapper:
    """
    Wrapper for multiple TimestampedVideoSource objects.
    """
    def __init__(self):
        self.sources = []
        self.frames = []
        self.timestamps = []
        self.save_timestamps = True
        self.num_sources = 0

    def add_camera(self, camera_number, dims=None):
        """
        Create VideoCapture object from camera and add it to the list
        of sources.

        :param camera_number: integer camera number
        :param dims: (width, height) as integer numbers of pixels
        """
        cu.validate_camera_input(camera_number)
        self.add_source(camera_number, dims)

    def add_file(self, filename, dims=None):
        """
        Create videoCapture object from file and add it to the list of sources.

        :param filename: a string containing a valid file path
        :param dims: (width, height) as integer numbers of pixels
        """
        vf.validate_is_file(filename)
        self.add_source(filename, dims)

    def add_source(self, camera_num_or_file, dims=None):
        """
         Add a video source (camera or file) to the list of sources.

        :param camera_num_or_file: either an integer camera number or filename
        :param dims: (width, height) as integer numbers of pixels
        """

        video_source = TimestampedVideoSource(camera_num_or_file, dims)
        self.sources.append(video_source)
        self.num_sources = len(self.sources)

    def are_all_sources_open(self):
        """
        Check all input sources are active/open.
        """
        for source in self.sources:
            if not source.isOpened():
                return False

        return True

    def release_all_sources(self):
        """
        Close all camera/file sources.
        """
        logging.info("Releasing video sources")
        for source in self.sources:
            source.release()

    def get_next_frames(self):
        """
        Do a grab() operation for each source,
        followed by a retrieve().
        """
        self.grab()
        self.retrieve()

    def grab(self):
        """
        Perform a grab() operation for each source
        """
        if self.are_all_sources_open():

            for source in self.sources:
                source.grab()



    def retrieve(self):
        """
        Perform a retrieve operation for each source.
        Should only be run after a grab() operation.

        :returns list of views on frames
        """
        self.frames = []
        for source in self.sources:
            source.retrieve()
            self.frames.append(source.frame)
        return self.frames
