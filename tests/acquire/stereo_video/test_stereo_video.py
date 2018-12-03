# coding=utf-8

import numpy as np
import cv2
import pytest
import sksurgeryimage.acquire.stereo_video as sv


def test_create_video_from_png():
    original = cv2.imread('tests/data/test-16x8-rgb.png')
    output_name = 'tests/output/test-16x8-rgb.avi'
    writer = cv2.VideoWriter(output_name,
                             cv2.VideoWriter_fourcc(*'DIVX'),
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


def test_interlaced_extract_original_images(interlaced_video_source):
    original = cv2.imread('tests/data/test-16x8-rgb.png')
    expected_even = cv2.imread('tests/data/test-16x8-rgb-even.png')
    expected_odd = cv2.imread('tests/data/test-16x8-rgb-odd.png')
    vs = interlaced_video_source
    vs.grab()
    vs.retrieve()

    vs.video_sources.frames[0] = original  # Hack for now, as colour space is changing !?!?

    even, odd = vs.get_images()
    assert even.shape[0] == original.shape[0] // 2
    assert even.shape[1] == original.shape[1]
    assert odd.shape[0] == original.shape[0] // 2
    assert odd.shape[1] == original.shape[1]
    np.testing.assert_array_equal(even, expected_even)
    np.testing.assert_array_equal(odd, expected_odd)


def test_interlaced_extract_scaled_images(interlaced_video_source):
    original = cv2.imread('tests/data/test-16x8-rgb.png')
    vs = interlaced_video_source
    vs.grab()
    vs.retrieve()
    even, odd = vs.get_scaled()
    assert even.shape[0] == original.shape[0]
    assert even.shape[1] == original.shape[1]
    assert odd.shape[0] == original.shape[0]
    assert odd.shape[1] == original.shape[1]


def test_vertically_stacked_extract_original_images(vertically_stacked_video_source):
    original = cv2.imread('tests/data/test-16x8-rgb.png')
    expected_top = cv2.imread('tests/data/test-16x8-rgb-top.png')
    expected_bottom = cv2.imread('tests/data/test-16x8-rgb-bottom.png')
    vs = vertically_stacked_video_source
    vs.grab()
    vs.retrieve()

    vs.video_sources.frames[0] = original  # Hack for now, as colour space is changing !?!?

    top, bottom = vs.get_images()
    assert top.shape[0] == original.shape[0] // 2
    assert bottom.shape[1] == original.shape[1]
    assert bottom.shape[0] == original.shape[0] // 2
    assert bottom.shape[1] == original.shape[1]
    np.testing.assert_array_equal(top, expected_top)
    np.testing.assert_array_equal(bottom, expected_bottom)
