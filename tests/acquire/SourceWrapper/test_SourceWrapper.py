import cv2
import numpy as np 
import pytest
from sksurgeryimage.acquire import SourceWrapper

def test_validate_camera_input(source_wrapper):

    try:
        valid_camera = 0
        assert source_wrapper.validate_camera_input(valid_camera)

    except IndexError:
        # No cameras are available
        True


def test_invalid_camera_input_throws_error(source_wrapper):

    invalid_camera = 'string.avi'
    with pytest.raises(TypeError):
        source_wrapper.validate_camera_input(invalid_camera)


def test_camera_does_not_exist_throws_error(source_wrapper):
    not_a_camera = 1234
    with pytest.raises(IndexError):
        source_wrapper.validate_camera_input(not_a_camera)


def test_validate_file_input(source_wrapper):

    file_that_exists = 'tox.ini'
    assert source_wrapper.validate_file_input(file_that_exists)


def test_invalid_file_input_throws_error(source_wrapper):

    invalid_filename = 1234
    with pytest.raises(ValueError):
        source_wrapper.validate_file_input(invalid_filename)


def test_add_source_from_file(source_wrapper):
    filename = 'tests/data/acquire/100x50_100_frames.avi'
    source_wrapper.add_file(filename)

    assert source_wrapper.frames[0].shape == (100, 50, 3)
    assert source_wrapper.are_all_sources_open()

    source_wrapper.release_all_sources()
    assert not source_wrapper.are_all_sources_open()

def test_add_source_from_camera(source_wrapper):
    """
    See if there is a camera available, if so run some tests.
    """
    try:
        camera_input = 0
        source_wrapper.add_camera(camera_input)
        assert source_wrapper.are_all_sources_open()

        source_wrapper.release_all_sources()
        assert not source_wrapper.are_all_sources_open()
    
    
    except IndexError:
        # No cameras availble
        return


def test_get_next_frames_from_file(source_wrapper):
    filename = 'tests/data/acquire/100x50_100_frames.avi'
    source_wrapper.add_source(filename)

    source_wrapper.get_next_frames()

    expected_frame = np.zeros((100,50,3), dtype=np.uint8)
    actual_frame = source_wrapper.frames[0]
    np.testing.assert_array_equal(expected_frame, actual_frame)


def test_do_timestamps_with_frame_update(source_wrapper):
    filename = 'tests/data/acquire/100x50_100_frames.avi'
    source_wrapper.add_source(filename)

    n_frames = 5
    for i in range(n_frames):
        source_wrapper.get_next_frames()

    assert len(source_wrapper.timestamps) == n_frames


def test_add_timestamps_from_two_sources(source_wrapper):
    # Pretend we have 2 sources connected
    source_wrapper.num_sources = 2
    
    # Source 0, 1st frame
    source_wrapper.add_timestamp_to_list(0)
    assert source_wrapper.timestamps[0].startswith('0,0')

    #source 1, 1st frame
    source_wrapper.add_timestamp_to_list(1)
    assert source_wrapper.timestamps[1].startswith('1,0')

    #source 2, 2nd frame
    source_wrapper.add_timestamp_to_list(0)
    assert source_wrapper.timestamps[2].startswith('0,1')

       












