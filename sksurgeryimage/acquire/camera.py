#coding=utf-8

""" Classes to handle capture of data from multiple camera sources at the same time """
import cv2
import numpy as np
import logging
import datetime

LOGGER = logging.getLogger(__name__)

class CameraWrapper():
    
    def __init__(self):
        self.cameras = []
        self.frames = []
        self.fps = 30
        self.do_timestamps = False
    
    def add_cameras(self, camera_inputs):
        """
        Take one (integer) or more (list of ints) camera inputs and
        open them.
        """

        if self.is_single_camera_input(camera_inputs):
            self.add_camera(camera_inputs)

        else:
            for camera_input in camera_inputs:
                self.add_camera(camera_input)

    @staticmethod
    def is_single_camera_input(camera_inputs):
        """ Check if input is a single value or a list"""
        if isinstance(camera_inputs, int):
            return True
        
        return False


    def add_camera(self, camera_input):
        """ Add a VideoCapture object to the cameras list"""
        
        LOGGER.info("Adding camera input: %d", camera_input)

        cam = cv2.VideoCapture(camera_input)
        self.cameras.append(cam)

        width = int(cam.get(3))
        height = int(cam.get(4))
        
        LOGGER.info("Video dimensions %s %s", width, height)

        empty_frame = np.empty((width, height, 3), dtype = np.uint8)
        self.frames.append(empty_frame)

        # Update the output video dimensions and internal array
        self.output_video_dimensions = self.get_output_video_dimensions()

        width, height = self.output_video_dimensions
        self.output_array = np.empty((height, width, 3), dtype = np.uint8)


    def get_output_video_dimensions(self):
            """ Set the video dimensions of the output video, combining the different inputs into one file"""
            #Side by side
            max_height = 0
            total_width = 0

            for camera in self.cameras:
                width = int(camera.get(3))
                height = int(camera.get(4))

                total_width += width

                if height > max_height:
                    max_height = height

            LOGGER.info("Output video dimensions: %d %d ", total_width, max_height)
            return (total_width, max_height)

    # TODO: Separate out the calculation of the dimensions into a separate method, for easier testing