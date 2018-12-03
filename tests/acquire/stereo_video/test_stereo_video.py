# coding=utf-8

import numpy as np
import cv2
import pytest
import sksurgeryimage.acquire.stereo_video as sv


def test_create_video_from_png():
    original = cv2.imread('tests/data/test-16x8-rgb.png')
    output_name = 'tests/output/test-16x8-rgb.avi'

    writer = cv2.VideoWriter(output_name,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             30,
                             (original.shape[1], original.shape[0]))
    writer.write(original)


def test_create_stereo_video_invalid_because_null_layout():
    with pytest.raises(ValueError):
        sv.StereoVideo(None, None, None)


def test_create_stereo_video_invalid_because_layout_wrong_value():
    with pytest.raises(ValueError):
        sv.StereoVideo("Banana", None, None)


def test_create_stereo_video_invalid_because_null_channel():
    with pytest.raises(ValueError):
        sv.StereoVideo(sv.StereoVideoLayouts.DUAL, None)


def test_create_stereo_video_invalid_because_too_few_channels():
    with pytest.raises(ValueError):
        sv.StereoVideo(sv.StereoVideoLayouts.INTERLACED, [])


def test_create_stereo_video_invalid_because_too_many_channels():
    with pytest.raises(ValueError):
        sv.StereoVideo(sv.StereoVideoLayouts.INTERLACED, [1, 2, 3])


def test_create_stereo_video_invalid_because_first_channel_not_correct_type():
    with pytest.raises(TypeError):
        sv.StereoVideo(sv.StereoVideoLayouts.INTERLACED, [np.ones((1, 1))])


def test_create_stereo_video_invalid_because_second_channel_not_correct_type():
    with pytest.raises(TypeError):
        sv.StereoVideo(sv.StereoVideoLayouts.DUAL,
                       ["tests/output/test-16x8-rgb.avi", np.ones((1, 1))])


def test_create_stereo_video_invalid_because_too_few_channels_for_dual():
    with pytest.raises(ValueError):
        sv.StereoVideo(sv.StereoVideoLayouts.DUAL, [0])


def test_set_camera_params_invalid_because_too_few_camera_matrices(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([], [])


def test_set_camera_params_invalid_because_too_many_camera_matrices(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3))], [])


def test_set_camera_params_invalid_because_too_few_distortion_matrices(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [])


def test_set_camera_params_invalid_because_too_many_distortion_matrices(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 5)), np.ones((1, 5)), np.ones((1, 5))])


def test_set_camera_params_invalid_because_camera_matrix_1_too_few_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((1, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_1_too_many_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((4, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_1_too_few_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 1)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_1_too_many_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 4)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_2_too_few_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((1, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_2_too_many_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((4, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_2_too_few_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 1)), np.ones((3, 1))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_2_too_many_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 4)), np.ones((3, 4))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_1_too_many_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((2, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_1_too_few_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 3)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_1_too_many_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 20)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_1_invalid_number_of_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 6)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_2_too_many_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((2, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_2_too_few_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 3))])


def test_set_camera_params_invalid_because_distortion_matrix_2_too_many_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 20))])


def test_set_camera_params_invalid_because_distortion_matrix_2_invalid_number_of_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_camera_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 7))])
