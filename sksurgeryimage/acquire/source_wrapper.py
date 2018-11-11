# coding=utf-8

"""
Module for video source acquisition.
Classes capture data from a video source into a numpy array.
"""

import logging
import datetime
import cv2
import numpy as np
import sksurgeryimage.utilities.camera_utilities as cu
import sksurgeryimage.utilities.utilities as u

LOGGER = logging.getLogger(__name__)


class VideoSourceWrapper:
    """
    Capture data from one or more camera/file sources.
    """
    def __init__(self):
        self.sources = []
        self.frames = []
        self.timestamps = []
        self.save_timestamps = False
        self.num_sources = 0

    def add_camera(self, camera_number, dims=None):
        """
        Create VideoCapture object from camera and add it to the list
        of sources.

        :param: dims is (width, height).
        """
        cu.validate_camera_input(camera_number)

        LOGGER.info("Adding camera input: %s", camera_number)
        self.add_source(camera_number, dims)

    def add_file(self, filename):
        """
        Create videoCapture object from file and add it to the list of sources.
        """
        u.validate_file_input(filename)

        LOGGER.info("Adding file input: %s", filename)
        self.add_source(filename)

    def add_source(self, source_num_or_file, dims=None):
        """
         Add a video source (camera or file) to the list of sources.

        :param: dims is (width, height).
        """

        video_source = cv2.VideoCapture(source_num_or_file)
        self.sources.append(video_source)

        if dims:
            width, height = dims
            video_source.set(3, width)
            video_source.set(4, height)

        else:
            width = int(video_source.get(3))
            height = int(video_source.get(4))

        LOGGER.info("Source dimensions %s %s", width, height)

        empty_frame = np.empty((height, width, 3), dtype=np.uint8)
        self.frames.append(empty_frame)
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
        Do a grab() operation for each source, timestamp (if wanted) and then
        retrieve() for each device.
        """
        # pylint: disable=unused-variable
        # 'ret' isn't used at the moment, but keep it for convention.

        if self.are_all_sources_open():

            for i, source in enumerate(self.sources):
                ret = source.grab()

                if self.save_timestamps:
                    self.add_timestamp_to_list(i)

            for i, source in enumerate(self.sources):
                ret, self.frames[i] = source.retrieve()

    def add_timestamp_to_list(self, source_number):
        """
        Get the current time and append a timestamp to the list of
        timestamps in format:
        source_num,frame_num,timestamp
        """
        now = datetime.datetime.now().isoformat()

        idx = len(self.timestamps)

        # If there is more than one video source, then we put one frame from
        #  each source in the list, before moving to next frame
        frame_num = idx // self.num_sources

        timestamp_entry = "{},{},{}".format(source_number, frame_num, now)
        self.timestamps.append(timestamp_entry)
