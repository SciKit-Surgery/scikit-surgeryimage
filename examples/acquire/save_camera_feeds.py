# Having some odd problem with import paths, this solves it for now
# TODO: Fix this properly
import os, sys
sys.path.append(os.getcwd())
sys.path.append('../scikit-surgeryimage')

import logging
from sksurgeryimage.acquire import utilities
from sksurgeryimage.acquire import video_writer, source_wrapper
import os
import sys
sys.path.append(os.getcwd())
###


logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def main():
    save_all_cameras_and_one_file()

def save_two_camera_feeds():
    """
    Save the feeds from two cameras.
    ""

    source_wrapper = 
def save_all_cameras_and_one_file():
    """
    Saves a camera feed from all attached cameras and a file
    """

    source_wrapper = SourceWrapper.VideoSourceWrapper()
    source_wrapper.show_acquired_frames = False
    source_wrapper.save_timestamps = True
    source_wrapper.add_file('tests/data/acquire/100x50_100_frames.avi')

    num_cameras = utilities.count_cameras()

    source_wrapper.add_camera(0)

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename = output_dir + '/test.avi'
    video_writer = VideoWriter.OneSourcePerFileWriter(base_filename)
    video_writer.set_frame_source(source_wrapper)

    video_writer.save_to_file(100)


if __name__ == "__main__":
    main()
