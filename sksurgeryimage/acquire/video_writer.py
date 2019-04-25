# coding=utf-8

""" Write stream of frames to file using OpenCV """

import logging
import os
import datetime
from queue import Queue
from threading import Thread
import cv2
import numpy as np

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
    def __init__(self, filename, fps=25, width=640, height=480, codec='MJPG'):

        self.set_filename(filename)

        fourcc = cv2.VideoWriter_fourcc(*codec)

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
        logging.debug("Closing VideoWriter")
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
            logging.debug("Creating directory: %s", directory)
            os.makedirs(directory)
            return True

        return False

    def write_frame(self, frame):
        """
        Write a frame to the output file.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame should be numpy array")

        logging.debug("Writing frame with dimensions: %i x %i",
                      frame.shape[1], frame.shape[0])

        self.video_writer.write(frame)



class TimestampedVideoWriter(VideoWriter):
    """
    Class to write images and timestamps to disk, inherits from VideoWriter.

    :param fps: Frames per second to save to disk.
    :param filename: Filename to save output video to.
                     Timestamp file is "filename + 'timestamps'"
    """
    def __init__(self, filename, fps=25, width=640,
                 height=480, codec='MJPG'):

        super(TimestampedVideoWriter, self).__init__(filename, fps,
                                                     width, height, codec)

        timestamp_filename = filename + '.timestamps'
        self.timestamp_file = open(timestamp_filename, 'w')
        self.default_timestamp_message = "NO_TIMESTAMP"


    def close(self):
        """ Close/release the output files for video and timestamps. """
        logging.debug("Closing video writer")
        self.video_writer.release()
        self.timestamp_file.close()
        logging.debug("Closing TimestampedVideoWriter.")

    def write_frame(self, frame, timestamp=None):
        """
        Write a frame and timestamp to the output files.
        If no timestamp provided, write a defualt value.
        :param frame: Image data
        :type frame: numpy array
        :param timestamp: Timestamp data
        :type timestamp: datetime.datetime object
        """
        super(TimestampedVideoWriter, self).write_frame(frame)

        if not timestamp:
            timestamp = self.default_timestamp_message
            self.timestamp_file.write(timestamp + '\n')
            return

        if not isinstance(timestamp, datetime.datetime):
            raise TypeError("Timestamp should be a datetime.datetimeobject")

        # Convert datetime object to string
        self.timestamp_file.write(timestamp.isoformat() + '\n')

class ThreadedTimestampedVideoWriter(TimestampedVideoWriter):
    """ TimestampedVideoWriter that can be run in a thread.
    Uses Queue.Queue() to store data, which is thread safe.

    Frames will be processed as they are added to the queue:

    threaded_vw = ThreadedTimestampedVideoWriter(file, fps, w, h)
    threaded_vw.start()

    threaded_vw.add_to_queue(frame, timestamp)
    threaded_vw.add_to_queue(frame, timestamp)
    threaded_vw.add_to_queue(frame, timestamp)

    threaded_vw.stop() """

    def __init__(self, filename, fps=25, width=640,
                 height=480, codec='MJPG'):

        super(ThreadedTimestampedVideoWriter, self).__init__(filename, fps,
                                                             width, height,
                                                             codec)
        self.started = False
        self.queue = Queue()

    def start(self):
        """ Start the thread running. """
        logging.debug("Starting ThreadedTimestampedVideoWriter thread")
        self.started = True
        Thread(target=self.run, args=()).start()
        return self

    def stop(self):
        """ Stop thread running. """
        logging.debug("Stopping ThreadedTimestampedVideoWriter thread")
        self.started = False

    def add_to_queue(self, frame, timestamp=None):
        """ Add a frame and a timestamp to the queue for writing.
        :param frame: Image frame
        :type frame: numpy array
        :param timestamp: Frame timestamp
        :type timestamp: datetime.datetime object """

        self.queue.put((frame, timestamp))

    def run(self):
        """ Write data from the queue to the output file(s). """

        # Write frames in the queue as they arrive
        while self.started:
            if not self.queue.empty():
                self.write_next_frame_and_timestamp()

        # Write any remaining frames in the queue
        while not self.queue.empty():
            self.write_next_frame_and_timestamp()

        self.close()

    def write_next_frame_and_timestamp(self):
        """ Get frame and timestamp from queue, then write to output. """
        logging.debug("Writing frame")
        frame, timestamp = self.queue.get()
        self.write_frame(frame, timestamp)
