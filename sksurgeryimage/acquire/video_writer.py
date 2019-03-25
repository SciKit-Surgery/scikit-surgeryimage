# coding=utf-8

""" Write stream of frames to file using OpenCV """

import logging
import os
import cv2

LOGGER = logging.getLogger(__name__)
#pylint:disable=useless-object-inheritance
class VideoWriter(object):
    """
    Class to write images to disk using cv2.VideoWriter.

    :param fps: Frames per second to save to disk.
    :param filename: Filename to save output video to.
    :param width: width of input frame
    :param height: height of input frame
    """
    def __init__(self, filename, fps, width, height):

        self.set_filename(filename)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

        self.video_writer = cv2.VideoWriter(
            filename, fourcc, fps, (width, height))

        logging.debug(
            "New VideoWriter. File:%s codec:%s FPS:%s Width:%s Height:%s",
            filename, fourcc, fps, width, height)

    def __del__(self):
        """ Call close method on deletion, in case not called manually. """
        self.close()

    def close(self):
        """ Close/release the output file for video. """
        self.video_writer.release()

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
            logging.info("Creating directory: %s", directory)
            os.makedirs(directory)
            return True

        return False

    def write_frame(self, frame):
        """
        Write a frame to the output file.
        """
        self.video_writer.write(frame)


class TimestampedVideoWriter(VideoWriter):
    """
    Class to write images and timestamps to disk, inherits from VideoWriter.

    :param fps: Frames per second to save to disk.
    :param filename: Filename to save output video to.
                     Timestamp file is "filename + 'timestamps'"
    """
    def __init__(self, filename, fps, width, height):
        super(TimestampedVideoWriter, self).__init__(filename, fps,
                                                     width, height)

        timestamp_filename = filename + '.timestamps'
        self.timestamp_file = open(timestamp_filename, 'w')

    def close(self):
        """ Close/release the output files for video and timestamps. """
        self.video_writer.release()
        self.timestamp_file.close()

    def write_frame(self, frame, timestamp):
        """
        Write a frame and timestamp to the output files.
        """
        self.video_writer.write(frame)
        self.timestamp_file.write(timestamp.isoformat() + '\n')
