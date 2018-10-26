### Having some odd problem with import paths, this solves it for now
### TODO: Fix this properly
import os, sys
sys.path.append(os.getcwd())
###

from sksurgeryimage.acquire import camera, utilities
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def main():
    save_camera_feeds()

def save_camera_feeds():
    """
    Saves a stacked feed from all connected cameras.
    """
    cam_wrapper = camera.CameraWrapper()
    cam_wrapper.do_timestamps = True
    cam_wrapper.stack_direction = "horizontal"

    camera_inputs = range(utilities.count_cameras()) 
    cam_wrapper.add_cameras(camera_inputs)

    output_file = 'test.avi'
    frames_to_grab = 100
    cam_wrapper.save_to_file(output_file, frames_to_grab)


if __name__ == "__main__":
    main()