#coding=utf-8

""" Classes to handle capture of data from multiple camera sources at the same time """

import cv2
import numpy as np
import logging
import datetime
import os

LOGGER = logging.getLogger(__name__)

class CameraWrapper():
    # TODO: Option to select stacking on/off
    # TODO: Something sensible to set fps
    def __init__(self):
        self.cameras = []
        self.frames = []
        self.fps = 30
        self.save_timestamps = False

        self.do_stacking = False
        self.stack_direction = "horizontal"
    

    def add_cameras(self, camera_inputs):
        """
        Take one (integer) or more (list of ints) camera inputs and
        open them.
        """

        if self.is_single_camera_input(camera_inputs):
            self.add_single_camera(camera_inputs)

        else:
            for camera_input in camera_inputs:
                self.add_single_camera(camera_input)


    @staticmethod
    def is_single_camera_input(camera_inputs):
        """ Check if input is a single value or a list"""
        if isinstance(camera_inputs, int):
            return True
        
        return False


    def add_single_camera(self, camera_input):
        """ Add a VideoCapture object to the cameras list"""
        
        LOGGER.info("Adding camera input: %d", camera_input)

        cam = cv2.VideoCapture(camera_input)
        self.cameras.append(cam)

        width = int(cam.get(3))
        height = int(cam.get(4))
        
        LOGGER.info("Video dimensions %s %s", width, height)

        empty_frame = np.empty((width, height, 3), dtype = np.uint8)
        self.frames.append(empty_frame)

        if self.do_stacking:
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
        self.output_array = np.empty((output_width, output_height, 3), dtype = np.uint8)

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

        
    def save_to_file(self, filename, frames_to_save = 9999999):
        #TODO: Refacotr & Tests

        self.create_video_writers(filename)
        # self.estimate_fps(filename)

        if self.save_timestamps:
            self.create_timestamp_array(frames_to_save)

        while self.are_all_cameras_open() and frames_to_save > 0:
            self.update_frames()

            if self.do_stacking:
                self.combine_frames()

            self.write_frames()

            frames_to_save -= 1

        self.release_cameras()
        self.release_video_writers()

        if self.save_timestamps:
            self.write_timestamps(filename)

    
    def create_video_writers(self, filename):
        """
        Create a CV2 Video Writer object to write to 'filename'
        """

        if not self.check_valid_filename(filename):
            return -1

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        if self.do_stacking:
            width, height = self.output_array.shape[:2]
            self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))

        else:
            self.video_writer = []
            filenames = self.generate_sequential_filenames(filename)

            for file, frame in zip(filenames, self.frames):
                width, height = frame.shape[:2]
                video_writer = cv2.VideoWriter(file, fourcc, self.fps, (width, height))
                self.video_writer.append(video_writer)

    
    def generate_sequential_filenames(self, filename):
        """
        Take a filename e.g. video.avi and generate a filename for each camera.
        video1.avi, video2.avi video3.avi etc.
        """

        filenames = []
        filename, extension = os.path.splitext(filename)

        for i, _ in enumerate(self.cameras):
            new_filename = "{}_{}{}".format(filename, i, extension)
            filenames.append(new_filename)

        return filenames

    
    @staticmethod
    def check_valid_filename(filename):
        if isinstance(filename, str):
            LOGGER.info("Output filename: %s", filename)
            return True

        LOGGER.info("Invalid filename passed")

        return False

    def create_timestamp_array(self, frames_to_save):
        """
        Create an array to hold timestamps (of unspecified type).
        Size should be n_frames * n_cameras.
        """

        self.timestamps = np.empty(frames_to_save * len(self.cameras), dtype=object)
        self.timestamp_idx = 0
    

    def are_all_cameras_open(self):
        """ Check all input cameras are active/open"""
        for camera in self.cameras:
            if not camera.isOpened():
                return False
        
        return True
    
    def release_cameras(self):
        """Close all camera objects"""
        logging.info("Releasing Cameras")
        for camera in self.cameras:
            camera.release()


    def release_video_writers(self):
        """
        Release all CV2 VideoWriter objects.
        """

        if self.do_stacking:
            self.video_writer.release()

        else:
            for video_writer in self.video_writer:
                video_writer.release()



    def update_frames(self):
        #TODO: Refactor & Tests
        """
        Do a grab() operation for each camera, timestamp (if wanted) and then 
        retrieve() for each device.
        """
    
        for camera in self.cameras:
            ret = camera.grab()

            if self.save_timestamps:
                now = datetime.datetime.now().isoformat()
                self.timestamps[self.timestamp_idx] = now
                self.timestamp_idx += 1

        for i, camera in enumerate(self.cameras):
           ret,  self.frames[i] = camera.retrieve()


        # if self.save_timestamps:
        #     for i, frame in enumerate(self.frames):
        #         annotate_text_to_frame(timestamps[i], frame)


    def combine_frames(self):
        #TODO: Refactor & Tests
        """Put all the camera frames in a single array"""

        #Side by side
        cumulative_width = 0
        for frame in self.frames:
            height, width = frame.shape[:2]
            logging.debug("Frame dims: %s x %s", width, height)

            x_start = cumulative_width
            x_end = cumulative_width + width
            y_start = 0
            y_end = height

            self.output_array[y_start:y_end, x_start:x_end, :] = frame

            cumulative_width += width

    def write_frames(self):
        """Write data to output file"""

        if self.do_stacking:
            self.video_writer.write(self.output_array)
            logging.debug("Writing frame with dims {}".format(self.output_array.shape))


        else:
            for i, frame in enumerate(self.frames):
                logging.debug("Writing frame with dims {}".format(frame.shape))
                self.video_writer[i].write(frame)
        

    def write_timestamps(self, filename):
        #TODO: Tests
        timestamp_file = filename + '.timestamps'
        LOGGER.info("Writing timestamps to %s", timestamp_file)

        with open(timestamp_file, "w") as text_file:
            for line in self.timestamps:
                text_file.write(line + '\n')