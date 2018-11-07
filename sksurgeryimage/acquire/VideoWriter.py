# coding=utf-8

""" Write stream of frames to file using OpenCV """

import cv2
import numpy as numpy
import logging
import datetime
import os

LOGGER = logging.getLogger(__name__)


class VideoWriterBase():

    def __init__(self, filename=None):

        if filename:
            self.set_filename(filename)

        self.fps = 30
        self.frame_source = None

        self.frames_to_save = 0

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writers = []

    def set_frame_source(self, frame_source):
        """ 
        Set the object's frame_source variable.
        """
        self.frame_source = frame_source

    def set_filename(self, filename):
        """
        Set the filename to write to.
        """

        if self.check_valid_filename(filename):
            self.filename = filename

    def check_valid_filename(self, filename):
        if isinstance(filename, str):
            return True

        raise ValueError('Invalid filename passed')

    def save_to_file(self, num_frames=None):

        if num_frames:
            self.frames_to_save = num_frames

        self.create_video_writers()

        logging.info("Saving {} frames to {}".format(
            self.frames_to_save, self.filename))

        while self.frames_to_save > 0:
            self.frame_source.get_next_frames()
            self.write_frame()
            self.frames_to_save -= 1

        self.release_video_writers()

        if self.frame_source.save_timestamps:
            self.write_timestamps()

    def create_video_writers(self):
        raise NotImplementedError('Should have implemented this method.')

    def release_video_writers(self):
        """ 
        Close all video writer objects
        """

        if len(self.video_writers) == 0:
            logging.info("No video writers to close")
            return

        for video_writer in self.video_writers:
            video_writer.release()

    def write_frame(self):
        """
        Write a frame to the output files.
        """
        for i, frame in enumerate(self.frame_source.frames):

            logging.debug("Writing frame with dims {}".format(frame.shape))
            self.video_writers[i].write(frame)

    def write_timestamps(self):
        """
        Write the timestamps from frame_source object to a file.
        """
        timestamp_file = self.filename + '.timestamps'
        logging.info("Writing timestamps to %s", timestamp_file)

        with open(timestamp_file, "w") as text_file:
            for line in self.frame_source.timestamps:
                text_file.write(line + '\n')


class OneSourcePerFileWriter(VideoWriterBase):
    """
    Class to writes to a separate video file for each input source.
    """

    def create_video_writers(self):
        """
        Create a VideoWriter object for each input source.
        """
        filenames = self.generate_sequential_filenames()
        logging.info("Saving to: %s", filenames)

        for filename, frame in zip(filenames, self.frame_source.frames):
            height, width = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                filename, self.fourcc, self.fps, (width, height))

            self.video_writers.append(video_writer)
            logging.debug("Created VideoWriter. Filename: {} codec: {} FPS:{} Width:{} Height:{}".format(
                filename, self.fourcc, self.fps, width, height))

    def generate_sequential_filenames(self):
        """
        Take a filename e.g. video.avi and generate a filename for each camera.
        video1.avi, video2.avi video3.avi etc.
        """
        filenames = []
        filename, extension = os.path.splitext(self.filename)

        for i, _ in enumerate(self.frame_source.frames):
            new_filename = "{}_{}{}".format(filename, i, extension)
            filenames.append(new_filename)

        return filenames


class StackedVideoWriter(VideoWriterBase):

    def create_video_writers(self):

        height, width = self.frame_source.get_dims()
        self.video_writer = cv2.VideoWriter(
            self.filename, self.fourcc, self.fps, (width, height))

        LOGGER.info("Saving to : %s", self.filename)
        logging.debug("Created VideoWriter. Filename: {} codec: {} FPS:{} Width:{} Height:{}".format(
            self.filename, self.fourcc, self.fps, width, height))
