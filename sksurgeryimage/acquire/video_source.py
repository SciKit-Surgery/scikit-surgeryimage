# coding=utf-8
#pylint:disable=no-name-in-module

"""
Module for video source acquisition.
Classes capture data from a video source into a numpy array.
"""

import logging
import datetime
import cv2
from cv2 import VideoCapture
import numpy as np
import sksurgerycore.utilities.validate_file as vf
import sksurgeryimage.utilities.camera_utilities as cu

LOGGER = logging.getLogger(__name__)

class TimestampedVideoCapture(VideoCapture):
    """
    Wrapper class around cv2.VideoCapture, to record a timestamp
    on image acquisition, and to optionally set the dimensions of
    the video capture.

    :param source_num_or_file: integer camera number or file path
    :param dims: optional (width, height) as a pair of integers
    """
    def __init__(self, source_num_or_file, dims=None):

        super(TimestampedVideoCapture, self).__init__(source_num_or_file)

        self.frame = None
        self.timestamp = None

        if not self.isOpened():
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

            self.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        else:
            width = int(self.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))

        LOGGER.info("Source dimensions %s %s", width, height)

    def grab(self):
        """
        Thin wrapper to call the cv2.VideoCapture grab function
        and update the timestamp.
        """
        LOGGER.debug("Grabbing from: %s", self.source_name)
        ret = super(TimestampedVideoCapture, self).grab()

        now = datetime.datetime.now().isoformat()
        self.timestamp = now

        return ret

    def read(self):
        """
        Call the cv2.VideoCapture read function and
        store the returned frame.
        """
        self.grab()
        ret, self.frame = self.retrieve()
        return ret, self.frame


class VideoSource:
    """
    Capture and store data from camera/file source.
    Augments the cv2.VideoCapture() to provide passing of
    camera dimensions in constructor, and storage of frame data.
    """
    def __init__(self, source_num_or_file, dims=None):
        """
        Constructs a VideoSource.

        :param source_num_or_file: integer camera number or file path
        :param dims: optional (width, height) as a pair of integers
        """
        self.source = cv2.VideoCapture(source_num_or_file)
        self.timestamps = []
        self.save_timestamps = False

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
        Call the cv2.VideoCapture grab function.
        """
        LOGGER.debug("Grabbing from: %s", self.source_name)
        self.ret = self.source.grab()

        if self.save_timestamps:
            self.add_timestamp_to_list()

        return self.ret

    def retrieve(self):
        """
        Call the cv2.VideoCapture retrieve function and
        store the returned frame.
        """
        LOGGER.debug("Retrieving from: %s", self.source_name)
        self.ret, self.frame = self.source.retrieve()
        return self.ret, self.frame

    def read(self):
        """
        Call the cv2.VideoCapture read function and
        store the returned frame.
        """
        self.ret, self.frame = self.source.read()
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


    def add_timestamp_to_list(self):
        """
        Get the current time and append a timestamp to the list of
        timestamps.
        """
        now = datetime.datetime.now().isoformat()
        self.timestamps.append(now)



class VideoSourceWrapper:
    """
    Wrapper for multiple VideoSource objects.
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

        video_source = VideoSource(camera_num_or_file, dims)
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
        and timestamp if required.
        """
        if self.are_all_sources_open():

            for i, source in enumerate(self.sources):
                source.grab()

                if self.save_timestamps:
                    self.add_timestamp_to_list(i)

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

    def add_timestamp_to_list(self, source_number):
        """
        Get the current time and append a timestamp to the list of
        timestamps in format:
        source_num,frame_num,timestamp
        """
        now = datetime.datetime.now().isoformat()

        idx = len(self.timestamps)

        # If there is more than one video source, then we put one frame from
        # each source in the list, before moving to next frame
        frame_num = idx // self.num_sources

        timestamp_entry = "{},{},{}".format(source_number, frame_num, now)
        self.timestamps.append(timestamp_entry)
