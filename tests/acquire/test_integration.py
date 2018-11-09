# coding=utf-8

"""
Tests hooking together various camera acquisition functions.
"""

import os
import cv2
import numpy as np
from sksurgeryimage.acquire import VideoWriter, SourceWrapper
from sksurgeryimage.utilities import utilities, camera_utilities


def test_save_a_file_and_all_cameras():
    """
    Saves a camera feed from all attached cameras and a file
    """

    source_wrapper = SourceWrapper.VideoSourceWrapper()
    input_file = 'tests/data/acquire/100x50_100_frames.avi'
    num_frames_in_input_file = 100

    source_wrapper.add_file(input_file)

    num_cameras = camera_utilities.count_cameras()

    for camera in range(num_cameras):
        source_wrapper.add_camera(camera)

    output_dir = 'tests/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = output_dir + '/test.avi'
    video_writer = VideoWriter.OneSourcePerFileWriter(base_filename)
    video_writer.set_frame_source(source_wrapper)

    video_writer.save_to_file(num_frames_in_input_file)

    output_file_name = 'test_0.avi'
    output_file_full_path = output_dir + '/' + output_file_name
    source_wrapper.release_all_sources()

    # The input and output files should be identical

    input_video = cv2.VideoCapture(input_file)
    output_video = cv2.VideoCapture(output_file_full_path)

    for i in range(num_frames_in_input_file):
        ret, frame_in = input_video.read()
        ret, frame_out = output_video.read()

        np.testing.assert_array_equal(frame_in, frame_out)
