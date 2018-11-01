
import os
import filecmp
from sksurgeryimage.acquire import VideoWriter, SourceWrapper
from sksurgeryimage.acquire import utilities


def test_save_a_file_and_all_cameras():
    """
    Saves a camera feed from all attached cameras and a file
    """

    source_wrapper = SourceWrapper.VideoSourceWrapper()
    input_file = 'tests/data/acquire/100x50_100_frames.avi'
    num_frames_in_input_file = 100

    source_wrapper.add_file(input_file)
    
    num_cameras = utilities.count_cameras()

    for camera in range(num_cameras):
        source_wrapper.add_camera(camera)

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_filename = output_dir + '/test.avi'
    video_writer = VideoWriter.OneSourcePerFileWriter(base_filename)
    video_writer.set_frame_source(source_wrapper)

    video_writer.save_to_file(num_frames_in_input_file)

    output_file_name = 'test_0.avi'
    output_file_full_path = output_dir + '/' + output_file_name

    # The input and output files should be identical
    assert filecmp.cmp(input_file, output_file_full_path)