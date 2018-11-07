import cv2
import numpy as np
import logging
import datetime
import os

LOGGER = logging.getLogger(__name__)


class VideoSourceWrapper():

    def __init__(self):
        self.sources = []
        self.frames = []

        self.timestamps = []
        self.save_timestamps = False

        self.show_acquired_frames = False

    def add_camera(self, camera_number, dims=None):
        """
         Create VideoCapture object from camera and add it to the list of sources.
         dims is (width, height).
         """

        self.validate_camera_input(camera_number)

        LOGGER.info("Adding camera input: {}".format(camera_number))
        self.add_source(camera_number, dims)

    def add_file(self, filename):
        """
        Create videoCapture object from file and add it to the list of sources
        """

        self.validate_file_input(filename)
        LOGGER.info("Adding file input: {}".format(filename))
        self.add_source(filename)

    def add_source(self, source_num_or_file, dims=None):
        """
         Add a video source (camera or file) to the list of sources.
        dims is (width, height).

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

    def validate_camera_input(self, camera_input):
        """
         Camera inputs should be an integer, throw error if not
          """
        if isinstance(camera_input, int):

            cam = cv2.VideoCapture(camera_input)
            if cam.isOpened():
                cam.release()
                return True

            raise IndexError(
                'No camera source exists with number: '.format(camera_input))

        raise TypeError('Integer expected for camera input')

    def validate_file_input(self, file_input):
        """
        Check if source file exists.
        """
        if os.path.isfile(file_input):
            return True

        raise ValueError('Input file does not exist')

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

        if self.are_all_sources_open():

            for i, source in enumerate(self.sources):
                ret = source.grab()

                if self.save_timestamps:
                    self.add_timestamp_to_list(i)

            for i, source in enumerate(self.sources):
                ret, self.frames[i] = source.retrieve()

        if self.show_acquired_frames:
            self.display_latest_frame()

    def display_latest_frame(self):
        """
        Show all of the frames using OpenCV.
        """
        for i, frame in enumerate(self.frames):
            frame_title = str(i)
            cv2.imshow(frame_title, frame)
            cv2.waitKey(1)

    def add_timestamp_to_list(self, source_number):
        """
        Get the current time and append a timestamp to the list of
        timestamps in format:
        source_num,frame_num,timestamp
        """
        now = datetime.datetime.now().isoformat()

        idx = len(self.timestamps)

        # If there is more than one video source, then we put one frame from each source
        # in the list, before moving to next frame
        frame_num = idx // self.num_sources

        timestamp_entry = "{},{},{}".format(source_number, frame_num, now)
        self.timestamps.append(timestamp_entry)
