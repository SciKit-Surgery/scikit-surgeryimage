# coding=utf-8

""" Write stream of frames to file using OpenCV """

import logging
import os
import cv2

LOGGER = logging.getLogger(__name__)


class VideoWriterBase:
    """
    Base Class for Video Writer
    """
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
            self.create_output_dir_if_needed()

    def check_valid_filename(self, filename):
        """
        Return true if filename is a string.
        """
        if isinstance(filename, str):
            return True

        raise ValueError('Invalid filename passed {}'.format(filename))

    def create_output_dir_if_needed(self):
        """
        Check if the directory specified in file path exists
        and create if not.
        """
        directory = os.path.dirname(self.filename)

        if directory and not os.path.exists(directory):
            LOGGER.info("Creating directory: %s", directory)
            os.makedirs(directory)
            return True

        return False

    def save_to_file(self, num_frames=None):
        """
        Acquire and write frames.
        """
        if num_frames:
            self.frames_to_save = num_frames

        self.create_video_writers()

        logging.info("Saving %s frames to %s",
                     self.frames_to_save, self.filename)

        while self.frames_to_save > 0:
            self.frame_source.get_next_frames()
            self.write_frame()
            self.frames_to_save -= 1

        self.release_video_writers()

        if self.frame_source.save_timestamps:
            self.write_timestamps()

    def create_video_writers(self):
        """
        Subclasses should implement a function to create
        one or more Open-CV VideoWriter objects.
        """
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
        for i, source in enumerate(self.frame_source.sources):

            frame = source.frame
            logging.debug("Writing frame with dims %s", frame.shape)
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

        for filename, source in zip(filenames, self.frame_source.sources):
            frame = source.frame
            height, width = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                filename, self.fourcc, self.fps, (width, height))

            self.video_writers.append(video_writer)
            logging.debug(
                "New VideoWriter. File:%s codec:%s FPS:%s Width:%s Height:%s",
                filename, self.fourcc, self.fps, width, height)

    def generate_sequential_filenames(self):
        """
        Take a filename e.g. video.avi and generate a filename for each camera.

        video1.avi, video2.avi video3.avi etc.
        """
        filenames = []

        filename, extension = os.path.splitext(self.filename)
        LOGGER.debug("Generating sequential filenames for: %s", filename)
        for i, _ in enumerate(self.frame_source.sources):
            new_filename = "{}_{}{}".format(filename, i, extension)
            filenames.append(new_filename)

        return filenames
