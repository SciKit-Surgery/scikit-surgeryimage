import pytest
from sksurgeryimage.acquire import video_writer as vw
import numpy as np
import os


class DummyFrameSource:
    """ Class to represent a frame_source,
    basically just a wrapper around a list of numpy arrays.
    """

    def __init__(self, all_dims):
        self.frames = self.generate_frames(all_dims)
        self.save_timestamps = False

        self.timestamps = ["1", "2", "3", "4"]

    def generate_frames(self, all_dims):
        """ Generate numpy arrays.
        Inputs: all_dims - list of dimensions for generated arrays 
        e.g. [(640, 480), (100, 100)] 
        """
        frames = []
        for dims in all_dims:

            dims_with_rgb = (dims[0], dims[1], 3)
            array = np.ones(dims_with_rgb, dtype=np.uint8)
            frames.append(array)

        return frames

    def get_next_frames(self):
        """ Mock function """
        return True


def setup_video_writer(num_sources):

    frame_dims = [(640, 480)]
    all_frame_dims = frame_dims * num_sources

    base_filename = 'output.avi'
    video_writer = vw.OneSourcePerFileWriter(base_filename)

    frame_source = DummyFrameSource(all_frame_dims)
    video_writer.frame_source = frame_source

    return video_writer


def delete_generated_files(video_writer):
    """ Delete any files created by the VideoWriter """

    video_writer.release_video_writers()

    for filename in video_writer.generate_sequential_filenames():
        if os.path.isfile(filename):
            os.remove(filename)


@pytest.fixture(scope="function")
def video_writer_single_source():
    """ Return a VideoWriter object with a single source """

    video_writer = setup_video_writer(1)
    yield video_writer
    delete_generated_files(video_writer)


@pytest.fixture(scope="function")
def video_writer_five_sources():
    """ Return a VideoWriter object with 5 sources """

    video_writer = setup_video_writer(5)
    yield video_writer
    delete_generated_files(video_writer)
