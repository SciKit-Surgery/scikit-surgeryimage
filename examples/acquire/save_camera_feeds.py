# Having some odd problem with import paths, this solves it for now
# TODO: Fix this properly
import os
import sys
sys.path.append(os.getcwd())
sys.path.append('../scikit-surgeryimage')

from sksurgeryimage.acquire import video_writer, source_wrapper
from sksurgeryimage.utilities import camera_utilities
import logging
import cv2

###

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


def main():
    save_all_cameras_and_one_file()


def save_all_cameras_and_one_file():
    """
    Saves a camera feed from all attached cameras and a file
    """

    sources = source_wrapper.VideoSourceWrapper()
    sources.add_file('tests/data/acquire/100x50_100_frames.avi')

    #num_cameras = camera_utilities.count_cameras()

    sources.add_camera(0)

    filename = 'outputs/test.avi'
    print(1)
    writer = video_writer.OneSourcePerFileWriter(filename)
    print(2)
    writer.set_frame_source(sources)
    print(3)
    writer.save_to_file(100)


if __name__ == "__main__":
    main()
