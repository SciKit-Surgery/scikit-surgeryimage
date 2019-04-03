# coding=utf-8

import os
import pytest
import mock
import numpy as np
import datetime
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


def test_timestamp_videowriter_creates_timestamp_file(tmpdir):
    filename = os.path.join(tmpdir.dirname, 'test.avi')
    vw.TimestampedVideoWriter(filename, fps, width, height)

    assert os.path.isfile(filename)
    assert os.path.isfile(filename + '.timestamps')

    # Video writer is tested further in test_integration


def test_invalid_data_types_raise_errors(tmpdir):
    fname = os.path.join(tmpdir.dirname, 'test_raises_error.avi')

    # These values don't matter
    fps = 25
    w = 100
    h = 100

    video_writer = vw.TimestampedVideoWriter(fname, fps, w, h)

    with pytest.raises(TypeError):
        video_writer.write_frame("not_np_array", "not_datetime")

    with pytest.raises(TypeError):
        video_writer.write_frame("not_np_array", datetime.datetime.now())
