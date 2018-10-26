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
        self.stack_direction = "horizontal"
    
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

        self.update_output_video_dimensions()
        

    def update_output_video_dimensions(self):
        """ 
        Store the dimensions of all input cameras, and then get the required
        size of the output frame.
        """

        frame_dims = []
        for camera in self.cameras:
            width = int(camera.get(3))
            height = int(camera.get(4))

            frame_dims.append([width, height])

        output_width, output_height = self.calculate_stacked_frame_dims(frame_dims, self.stack_direction)
        self.output_array = np.empty((output_height, output_width, 3), dtype = np.uint8)

        LOGGER.info("Output video dimensions: %d %d ", output_width, output_height)

 
    @staticmethod
    def calculate_stacked_frame_dims(input_frame_dims, stack_direction = "horizontal"):
        """
        Calculate the required frame size to fit all of the input frame in.
        Inputs
        input_frame_dims: list of frame dimensions
        stack_direction: can be "horiziontal", "h", "vertical", "v". Default is horizontal.
        """

        if stack_direction.lower().startswith("h"):
            horizontal = True
        
        elif stack_direction.lower().startswith("v"):
            horizontal = False
        
        else:
           raise ValueError('Invalid stacking direction specified.')

        total_width  = 0
        total_height = 0
        max_width = 0
        max_height = 0

        for width, height in input_frame_dims:
            total_height += height
            total_width += width

            if height > max_height:
                max_height = height

            if width > max_width:
                max_width = width

        if horizontal:
            return (total_width, max_height)

        # Otherwise vertical stacking
        return (max_width, total_height)