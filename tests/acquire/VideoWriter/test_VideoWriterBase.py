import pytest
import numpy as np
import os
from sksurgeryimage.acquire import VideoWriter


def test_set_filename():
    """ Check a valid filename is accepted. """
    valid_filename = 'file.txt'
    video_writer = VideoWriter.VideoWriterBase(valid_filename)

    assert video_writer.check_valid_filename(valid_filename)
    assert video_writer.filename == valid_filename


def test_invalid_filename_raises_ValueError():
    """ Check an invalid filename (i.e. not a string) raises an error """
    invalid_name = 1
    with pytest.raises(ValueError):
        video_writer = VideoWriter.VideoWriterBase(invalid_name)


def test_release_video_writers_when_none_available():
    """ Check that calling release_video_writers when none are open returns False """
    valid_filename = 'file.txt'
    video_writer = VideoWriter.VideoWriterBase(valid_filename)
    assert not video_writer.release_video_writers()


def test_write_frame_throws_no_error(video_writer_five_sources):
    """ Check that the write_frame function runs without an error """
    video_writer = video_writer_five_sources
    video_writer.create_video_writers()

    video_writer.write_frame()


def test_save_to_file_throws_no_error(video_writer_five_sources):
    """ Check that save_to_file runs without an error """
    video_writer = video_writer_five_sources

    num_frames = 100
    video_writer.save_to_file(100)


def test_set_frame_source(video_writer_single_source):

    video_writer = video_writer_single_source
    frame_source = video_writer.frame_source
    video_writer.set_frame_source(None)

    assert video_writer.frame_source == None

    video_writer.set_frame_source(frame_source)
    assert video_writer.frame_source == frame_source


def test_write_timestamps_throws_no_error(video_writer_single_source):
    video_writer = video_writer_single_source
    video_writer.write_timestamps()
