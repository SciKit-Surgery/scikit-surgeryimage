import pytest
import numpy as np
import datetime
import sksurgerycore.utilities.validate_file as vf
import sksurgeryimage.utilities.camera_utilities as cu
import sksurgeryimage.utilities.utilities as u


def test_validate_camera_input(video_source_wrapper):

    try:
        valid_camera = 0
        assert cu.validate_camera_input(valid_camera)

    except IndexError:
        # No cameras are available
        True


def test_invalid_camera_input_throws_error(video_source_wrapper):

    invalid_camera = 'string.avi'
    with pytest.raises(TypeError):
        cu.validate_camera_input(invalid_camera)


def test_camera_does_not_exist_throws_error(video_source_wrapper):
    not_a_camera = 1234
    with pytest.raises(IndexError):
        cu.validate_camera_input(not_a_camera)


def test_validate_file_input(video_source_wrapper):

    file_that_exists = 'tox.ini'
    assert vf.validate_is_file(file_that_exists)


def test_invalid_file_input_throws_error(video_source_wrapper):

    invalid_filename = '1234'
    with pytest.raises(ValueError):
        vf.validate_is_file(invalid_filename)


def test_add_source_from_file(video_source_wrapper):
    filename = 'tests/data/acquire/100x50_100_frames.avi'
    video_source_wrapper.add_file(filename)

    assert video_source_wrapper.sources[0].frame.shape == (100, 50, 3)
    assert video_source_wrapper.are_all_sources_open()

    video_source_wrapper.release_all_sources()
    assert not video_source_wrapper.are_all_sources_open()


def test_add_source_from_camera(video_source_wrapper):
    """
    See if there is a camera available, if so run some tests.
    """
    try:
        camera_input = 0
        video_source_wrapper.add_camera(camera_input)
        assert video_source_wrapper.are_all_sources_open()

        video_source_wrapper.release_all_sources()
        assert not video_source_wrapper.are_all_sources_open()

    except IndexError:
        # No cameras availble
        return


def test_add_source_from_invalid_camera(video_source_wrapper):
    camera_input = -1
    with pytest.raises(IndexError):
        video_source_wrapper.add_camera(camera_input)


def test_add_source_from_camera_custom_dimensions(video_source_wrapper):
    """
    Add a camera and pass in custom dimensions to cv2.VideoCapture.
    """
    try:
        camera_input = 0
        custom_dims = [320, 240]  # default is 640 x 480
        video_source_wrapper.add_camera(camera_input, custom_dims)

        expected_output_dims = (240, 320, 3)
        assert video_source_wrapper.frames[0].shape == expected_output_dims

    except IndexError:
        return

def test_add_source_from_camera_invalid_dimensions(video_source_wrapper):
    """
    Add a camera and pass in custom dimensions to cv2.VideoCapture.
    """
    try:
        camera_input = 0
        custom_dims = [100, 24]  # default is 640 x 480

        with pytest.raises(ValueError):
            video_source_wrapper.add_camera(camera_input, custom_dims)

    except IndexError:
        return

def test_add_source_from_file_invalid_dims(video_source_wrapper):
    camera_input = 'tests/data/acquire/100x50_100_frames.avi'

    custom_dims = ["happy", "birthday"]
    with pytest.raises(TypeError):
        video_source_wrapper.add_file(camera_input, custom_dims)

    custom_dims = [240, "birthday"]
    with pytest.raises(TypeError):
        video_source_wrapper.add_file(camera_input, custom_dims)

    custom_dims = [0, 320]
    with pytest.raises(ValueError):
        video_source_wrapper.add_file(camera_input, custom_dims)

    custom_dims = [240, 0]
    with pytest.raises(ValueError):
        video_source_wrapper.add_file(camera_input, custom_dims)


def test_get_next_frames_from_file(video_source_wrapper):
    filename = 'tests/data/acquire/100x50_100_frames.avi'
    video_source_wrapper.add_source(filename)
    video_source_wrapper.get_next_frames()

    expected_frame = np.zeros((100, 50, 3), dtype=np.uint8)
    actual_frame = video_source_wrapper.sources[0].frame
    np.testing.assert_array_equal(expected_frame, actual_frame)

    # Same test, but use .read() rather than separate .grab() and .retrieve()
    video_source_wrapper.sources[0].read()
    actual_frame = video_source_wrapper.sources[0].frame
    np.testing.assert_array_equal(expected_frame, actual_frame)

def test_do_timestamps_in_source(video_source_wrapper):
    """ Get the video source to save a timestamp"""
    filename = 'tests/data/acquire/100x50_100_frames.avi'
    video_source_wrapper.add_file(filename)
    source = video_source_wrapper.sources[0]

    before_time = datetime.datetime.now()
    source.grab()
   
    #Time difference between the two calls to datetime.now() should be small
    time_diff = source.timestamp - before_time
    assert time_diff.seconds == 0
    ten_ms_in_us = 10000
    assert time_diff.microseconds < ten_ms_in_us


