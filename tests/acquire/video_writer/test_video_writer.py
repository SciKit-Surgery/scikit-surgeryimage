# coding=utf-8

import os
import pytest
import mock
from sksurgeryimage.acquire import video_writer as vw

fps = 25
width, height = (640, 480)

def test_videowriter_invalid_filename_raises_error():
    invalid_filename = 1234

    with pytest.raises(ValueError):
        vw.VideoWriter(invalid_filename, fps, width, height)

@mock.patch('os.makedirs')
def test_videowriter_create_dir(mocked_makedirs):
    filename = 'new_dir/file.avi'
    vw.VideoWriter(filename, fps, width, height)
    mocked_makedirs.assert_called_with('new_dir')

@pytest.fixture()
def create_timestamped_videowriter_output_files():
    filename = 'tests/generated_data/test.avi'
    vw.TimestampedVideoWriter(filename, fps, width, height)

    yield

    os.remove(filename)
    os.remove(filename + '.timestamps')
    os.rmdir('tests/generated_data')

def test_timestamp_videowriter_creates_timestamp_file(create_timestamped_videowriter_output_files):
    filename = 'tests/generated_data/test.avi'
    vw.TimestampedVideoWriter(filename, fps, width, height)

    assert os.path.isfile(filename)
    assert os.path.isfile(filename + '.timestamps')

    # Video writer is tested further in test_integration