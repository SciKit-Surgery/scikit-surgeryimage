# coding=utf-8

"""
Tests hooking together various camera acquisition functions.
"""
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../scikit-surgeryimage')
import os
import cv2
import numpy as np
from sksurgeryimage.acquire import video_writer, video_source
from sksurgeryimage.utilities import camera_utilities


def test_save_a_file_and_all_cameras():
    """
    Saves a camera feed from all attached cameras and a file
    """

    sw = video_source.VideoSourceWrapper()
    input_file = 'tests/data/acquire/100x50_100_frames.avi'
    num_frames_in_input_file = 100

    sw.add_file(input_file)

    num_cameras = camera_utilities.count_cameras()

    for camera in range(num_cameras):
        sw.add_camera(camera)
    
    output_dir = 'tests/output/acquire/'

    fps = 25

    # Create video writer for file input
    output_for_video_file = output_dir + 'video_from_file.avi'
    width, height  = (sw.sources[0].frame.shape[1],  sw.sources[0].frame.shape[0])
    video_writer_from_file = video_writer.TimestampedVideoWriter(output_for_video_file, fps, width, height)

    # Create video writer(s) for all camera inputs
    cam_writers = []
    for i in range(num_cameras):
        fname = output_dir + 'camera' + str(i) + '.avi'
        width, height = (sw.sources[i+1].frame.shape[1],  sw.sources[i+1].frame.shape[0])
        cam_writers.append(video_writer.TimestampedVideoWriter(fname, fps, width, height))

    ###############################
    # Capture and write some frames
    num_frames = 100
    for i in range(num_frames):
        sw.get_next_frames()
        frame = sw.sources[0].frame
        timestamp = sw.sources[0].timestamp
        video_writer_from_file.write_frame(frame, timestamp)

        for j in range(num_cameras):
            frame = sw.sources[j+1].frame
            timestamp = sw.sources[j+1].timestamp
            cam_writers[j].write_frame(frame, timestamp)

    # Close sources and writers
    sw.release_all_sources()

    video_writer_from_file.close()
    for i in range(num_cameras):
        cam_writers[i].close()

    ########################
    # Do the actual test
    # The input and output files should be identical

    input_video = cv2.VideoCapture(input_file)
    output_video = cv2.VideoCapture(output_for_video_file)

    for i in range(num_frames_in_input_file):
        ret, frame_in = input_video.read()
        ret, frame_out = output_video.read()
        np.testing.assert_array_equal(frame_in, frame_out)
    
    input_video.release()
    output_video.release()

def test_save_file_using_threaded_video_writer():
    """
    Saves a video feed  read from a file using ThreadedTimestampedVideoWriter.
    """

    sw = video_source.VideoSourceWrapper()
    input_file = 'tests/data/acquire/100x50_100_frames.avi'
    num_frames_in_input_file = 100

    sw.add_file(input_file)

    output_dir = 'tests/output/acquire/'

    # Create video writer for file input
    output_for_video_file = output_dir + 'threaded_video_writer_from_file.avi'
    fps = 25
    width, height  = (sw.sources[0].frame.shape[1],  sw.sources[0].frame.shape[0])
   
    threaded_vw = \
        video_writer.ThreadedTimestampedVideoWriter(output_for_video_file,
                                                    fps, width, height)

    threaded_vw.start()

    ###############################
    # Capture and write some frames
    for i in range(num_frames_in_input_file):
        sw.get_next_frames()
        frame = sw.sources[0].frame
        timestamp = sw.sources[0].timestamp
        threaded_vw.add_to_queue(frame, timestamp)

    # Close sources and writers
    threaded_vw.stop()
    sw.release_all_sources()
    
    # Wait for remaining frames to be written
    while not threaded_vw.queue.empty():
        True

    ########################
    # Do the actual test
    # The input and output files should be identical

    input_video = cv2.VideoCapture(input_file)
    output_video = cv2.VideoCapture(output_for_video_file)

    for i in range(num_frames_in_input_file):
        ret, frame_in = input_video.read()
        ret, frame_out = output_video.read()
        np.testing.assert_array_equal(frame_in, frame_out)
    
    input_video.release()
    output_video.release()