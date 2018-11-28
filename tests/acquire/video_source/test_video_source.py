import pytest
import numpy as np
import sksurgeryimage.utilities.camera_utilities as cu
import sksurgeryimage.utilities.utilities as u


def test_validate_camera_input(video_source):

    try:
        valid_camera = 0
        assert cu.validate_camera_input(valid_camera)

    except IndexError:
        # No cameras are available
        True


def test_invalid_camera_input_throws_error(video_source):

    invalid_camera = 'string.avi'
    with pytest.raises(TypeError):
        cu.validate_camera_input(invalid_camera)


def test_camera_does_not_exist_throws_error(video_source):
    not_a_camera = 1234
    with pytest.raises(IndexError):
        cu.validate_camera_input(not_a_camera)


def test_validate_file_input(video_source):

    file_that_exists = 'tox.ini'
    assert u.validate_file_input(file_that_exists)


def test_invalid_file_input_throws_error(video_source):

    invalid_filename = '1234'
    with pytest.raises(ValueError):
        u.validate_file_input(invalid_filename)


def test_add_source_from_file(video_source):
    filename = 'tests/data/acquire/100x50_100_frames.avi'
    video_source.add_file(filename)

    assert video_source.sources[0].frame.shape == (100, 50, 3)
    assert video_source.are_all_sources_open()

    video_source.release_all_sources()
    assert not video_source.are_all_sources_open()


def test_add_source_from_camera(video_source):
    """
    See if there is a camera available, if so run some tests.
    """
    try:
        camera_input = 0
        video_source.add_camera(camera_input)
        assert video_source.are_all_sources_open()

        video_source.release_all_sources()
        assert not video_source.are_all_sources_open()

    except IndexError:
        # No cameras availble
        return


def test_add_source_from_invalid_camera(video_source):
    camera_input = -1
    with pytest.raises(IndexError):
        video_source.add_camera(camera_input)


def test_add_source_from_camera_custom_dimensions(video_source):
    """
    Add a camera and pass in custom dimensions to cv2.VideoCapture.
    """
    try:
        camera_input = 0
        custom_dims = [320, 240]  # default is 640 x 480
        video_source.add_camera(camera_input, custom_dims)

        expected_output_dims = (240, 320, 3)
        assert video_source.frames[0].shape == expected_output_dims

    except IndexError:
        return


def test_add_source_from_camera_invalid_dims(video_source):
    camera_input = 0

    custom_dims = ["happy", "birthday"]
    with pytest.raises(TypeError):
        video_source.add_camera(camera_input, custom_dims)

    custom_dims = [240, "birthday"]
    with pytest.raises(TypeError):
        video_source.add_camera(camera_input, custom_dims)

    custom_dims = [0, 320]
    with pytest.raises(ValueError):
        video_source.add_camera(camera_input, custom_dims)

    custom_dims = [240, 0]
    with pytest.raises(ValueError):
        video_source.add_camera(camera_input, custom_dims)


def test_get_next_frames_from_file(video_source):
    filename = 'tests/data/acquire/100x50_100_frames.avi'
    video_source.add_source(filename)
    video_source.get_next_frames()

    expected_frame = np.zeros((100, 50, 3), dtype=np.uint8)
    actual_frame = video_source.sources[0].frame
    np.testing.assert_array_equal(expected_frame, actual_frame)


def test_do_timestamps_with_frame_update(video_source):
    filename = 'tests/data/acquire/100x50_100_frames.avi'
    video_source.add_file(filename)
    video_source.save_timestamps = True

    n_frames = 5
    for i in range(n_frames):
        video_source.get_next_frames()

    assert len(video_source.timestamps) == n_frames


def test_add_timestamps_from_two_sources(video_source):
    # Pretend we have 2 sources connected
    video_source.num_sources = 2
    video_source.save_timestamps = True

    # Source 0, 1st frame
    video_source.add_timestamp_to_list(0)
    assert video_source.timestamps[0].startswith('0,0')

    # source 1, 1st frame
    video_source.add_timestamp_to_list(1)
    assert video_source.timestamps[1].startswith('1,0')

    # source 2, 2nd frame
    video_source.add_timestamp_to_list(0)
    assert video_source.timestamps[2].startswith('0,1')
