# coding=utf-8

import numpy as np
import cv2
import pytest
import sksurgeryimage.acquire.stereo_video as sv
from sksurgeryimage.utilities.utilities import are_similar


def test_create_video_from_png():
    original = cv2.imread('tests/data/processing/test-16x8-rgb.png')
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


def test_create_stereo_video_invalid_because_null_channel1():
    with pytest.raises(ValueError):
        sv.StereoVideo(sv.StereoVideoLayouts.DUAL, [None, 0])


def test_create_stereo_video_invalid_because_null_channel2():
    with pytest.raises(ValueError):
        sv.StereoVideo(sv.StereoVideoLayouts.DUAL, [0, None])


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
                       ["tests/data/acquire/test-16x8-rgb.avi", np.ones((1, 1))])


def test_create_stereo_video_invalid_because_too_few_channels_for_dual():
    with pytest.raises(ValueError):
        sv.StereoVideo(sv.StereoVideoLayouts.DUAL, [0])


def test_create_stereo_video_invalid_because_width_invalid_type():
    with pytest.raises(TypeError):
        sv.StereoVideo(sv.StereoVideoLayouts.INTERLACED,
                       ["tests/data/acquire/test-16x8-rgb.avi"],
                       ("a", "b")
                       )


def test_create_stereo_video_invalid_because_height_invalid_type():
    with pytest.raises(TypeError):
        sv.StereoVideo(sv.StereoVideoLayouts.INTERLACED,
                       ["tests/data/acquire/test-16x8-rgb.avi"],
                       (1, "b")
                       )


def test_create_stereo_video_invalid_because_width_too_low():
    with pytest.raises(ValueError):
        sv.StereoVideo(sv.StereoVideoLayouts.INTERLACED,
                       ["tests/data/acquire/test-16x8-rgb.avi"],
                       (-1, 1)
                       )


def test_create_stereo_video_invalid_because_height_too_low():
    with pytest.raises(ValueError):
        sv.StereoVideo(sv.StereoVideoLayouts.INTERLACED,
                       ["tests/data/acquire/test-16x8-rgb.avi"],
                       (1, -1)
                       )


def test_set_camera_params_invalid_because_too_few_camera_matrices(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([], [])


def test_set_camera_params_invalid_because_too_many_camera_matrices(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3)), np.ones((3, 3))], [])


def test_set_camera_params_invalid_because_too_few_distortion_matrices(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [])


def test_set_camera_params_invalid_because_too_many_distortion_matrices(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 5)), np.ones((1, 5)), np.ones((1, 5))])


def test_set_camera_params_invalid_because_camera_matrix_1_too_few_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((1, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_1_too_many_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((4, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_1_too_few_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 1)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_1_too_many_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 4)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_2_too_few_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((1, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_2_too_many_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((4, 3))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_2_too_few_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 1)), np.ones((3, 1))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_camera_matrix_2_too_many_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 4)), np.ones((3, 4))], [np.ones((1, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_1_too_many_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((2, 4)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_1_too_few_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 3)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_1_too_many_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 20)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_1_invalid_number_of_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 6)), np.ones((1, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_2_too_many_rows(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((2, 4))])


def test_set_camera_params_invalid_because_distortion_matrix_2_too_few_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 3))])


def test_set_camera_params_invalid_because_distortion_matrix_2_too_many_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 20))])


def test_set_camera_params_invalid_because_distortion_matrix_2_invalid_number_of_columns(interlaced_video_source):
    with pytest.raises(ValueError):
        interlaced_video_source.set_intrinsic_parameters([np.ones((3, 3)), np.ones((3, 3))], [np.ones((1, 4)), np.ones((1, 7))])


def test_set_camera_params_invalid_because_not_rotation_matrix(interlaced_video_source):
    with pytest.raises(TypeError):
        interlaced_video_source.set_extrinsic_parameters(1,2);


def test_interlaced_invalid_to_extract_data_before_calling_grab_and_retrieve(interlaced_video_source):
    vs = interlaced_video_source
    with pytest.raises(RuntimeError):
        vs.get_images()


def test_interlaced_extract_original_images(interlaced_video_source):
    original = cv2.imread('tests/data/processing/test-16x8-rgb.png')
    expected_even = cv2.imread('tests/data/processing/test-16x8-rgb-even.png')
    expected_odd = cv2.imread('tests/data/processing/test-16x8-rgb-odd.png')
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
    original = cv2.imread('tests/data/processing/test-16x8-rgb.png')
    vs = interlaced_video_source
    vs.grab()
    vs.retrieve()

    vs.video_sources.frames[0] = original  # Hack for now, as colour space is changing !?!?

    even, odd = vs.get_scaled()
    assert even.shape[0] == original.shape[0]
    assert even.shape[1] == original.shape[1]
    assert odd.shape[0] == original.shape[0]
    assert odd.shape[1] == original.shape[1]


def test_vertically_stacked_extract_original_images(vertically_stacked_video_source):
    original = cv2.imread('tests/data/processing/test-16x8-rgb.png')
    expected_top = cv2.imread('tests/data/processing/test-16x8-rgb-top.png')
    expected_bottom = cv2.imread('tests/data/processing/test-16x8-rgb-bottom.png')
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


def test_opencv_example_stereo_distortion_correction_and_rectification(two_channel_video_source):
    expected_original_left = cv2.imread('tests/data/calib-opencv/left01.jpg')
    expected_original_right = cv2.imread('tests/data/calib-opencv/right01.jpg')
    expected_undistorted_left = cv2.imread('tests/data/calib-opencv/left_cv_undistorted_4.1.1.26.png')
    expected_undistorted_right = cv2.imread('tests/data/calib-opencv/right_cv_undistorted_4.1.1.26.png')
    expected_rectified_left = cv2.imread('tests/data/calib-opencv/left_cv_rectified_4.1.1.26.png')
    expected_rectified_right = cv2.imread('tests/data/calib-opencv/right_cv_rectified_4.1.1.26.png')
    fs_li = cv2.FileStorage('tests/data/calib-opencv/calib.left.intrinsic.xml', cv2.FILE_STORAGE_READ)
    fs_ld = cv2.FileStorage('tests/data/calib-opencv/calib.left.distortion.xml', cv2.FILE_STORAGE_READ)
    fs_ri = cv2.FileStorage('tests/data/calib-opencv/calib.right.intrinsic.xml', cv2.FILE_STORAGE_READ)
    fs_rd = cv2.FileStorage('tests/data/calib-opencv/calib.right.distortion.xml', cv2.FILE_STORAGE_READ)
    fs_r = cv2.FileStorage('tests/data/calib-opencv/calib.r2l.rotation.xml', cv2.FILE_STORAGE_READ)
    fs_t = cv2.FileStorage('tests/data/calib-opencv/calib.r2l.translation.xml', cv2.FILE_STORAGE_READ)

    li = fs_li.getNode("calib_left_intrinsic").mat()
    ld = fs_ld.getNode("calib_left_distortion").mat()
    ri = fs_ri.getNode("calib_right_intrinsic").mat()
    rd = fs_rd.getNode("calib_right_distortion").mat()
    r = fs_r.getNode("calib_r2l_rotation").mat()
    t = fs_t.getNode("calib_r2l_translation").mat()

    vs = two_channel_video_source
    vs.grab()
    vs.retrieve()

    # Should fail, as we haven't set intrinsics yet.
    with pytest.raises(ValueError):
        vs.get_undistorted()

    vs.video_sources.frames[0] = expected_original_left  # Hack for now, as colour space is changing !?!?
    vs.video_sources.frames[1] = expected_original_right  # Hack for now, as colour space is changing !?!?

    vs.set_intrinsic_parameters([li, ri], [ld, rd])
    left, right = vs.get_undistorted()

    assert are_similar(left, expected_undistorted_left,
            threshold = 0.995, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.005)
    assert are_similar(right, expected_undistorted_right,
            threshold = 0.995, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.005)

    vs.set_extrinsic_parameters(r, t, (int(vs.video_sources.frames[0].shape[1]),
                                       int(vs.video_sources.frames[0].shape[0])))
    rectified_left, rectified_right = vs.get_rectified()

    assert are_similar(rectified_left, expected_rectified_left,
            threshold = 0.995, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.005)
    assert are_similar(rectified_right, expected_rectified_right,
            threshold = 0.995, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.005)

    vs.release()  # Just ensuring code doesn't crash


def test_ucl_example_stereo_distortion_correction_and_rectification(two_channel_ucl_video_source):
    expected_original_left = cv2.imread('tests/data/calib-ucl-chessboard/leftImage.png')
    expected_original_right = cv2.imread('tests/data/calib-ucl-chessboard/rightImage.png')
    expected_undistorted_left = cv2.imread('tests/data/calib-ucl-chessboard/left_ucl_undistorted_4.1.1.26.png')
    expected_undistorted_right = cv2.imread('tests/data/calib-ucl-chessboard/right_ucl_undistorted_4.1.1.26.png')
    expected_rectified_left = cv2.imread('tests/data/calib-ucl-chessboard/left_ucl_rectified_4.1.1.26.png')
    expected_rectified_right = cv2.imread('tests/data/calib-ucl-chessboard/right_ucl_rectified_4.1.1.26.png')
    fs_li = cv2.FileStorage('tests/data/calib-ucl-chessboard/calib.left.intrinsic.xml', cv2.FILE_STORAGE_READ)
    fs_ld = cv2.FileStorage('tests/data/calib-ucl-chessboard/calib.left.distortion.xml', cv2.FILE_STORAGE_READ)
    fs_ri = cv2.FileStorage('tests/data/calib-ucl-chessboard/calib.right.intrinsic.xml', cv2.FILE_STORAGE_READ)
    fs_rd = cv2.FileStorage('tests/data/calib-ucl-chessboard/calib.right.distortion.xml', cv2.FILE_STORAGE_READ)
    fs_r = cv2.FileStorage('tests/data/calib-ucl-chessboard/calib.r2l.rotation.xml', cv2.FILE_STORAGE_READ)
    fs_t = cv2.FileStorage('tests/data/calib-ucl-chessboard/calib.r2l.translation.xml', cv2.FILE_STORAGE_READ)

    li = fs_li.getNode("calib_left_intrinsic").mat()
    ld = fs_ld.getNode("calib_left_distortion").mat()
    ri = fs_ri.getNode("calib_right_intrinsic").mat()
    rd = fs_rd.getNode("calib_right_distortion").mat()
    r = fs_r.getNode("calib_r2l_rotation").mat()
    t = fs_t.getNode("calib_r2l_translation").mat()

    vs = two_channel_ucl_video_source
    vs.set_intrinsic_parameters([li, ri], [ld, rd])
    vs.grab()
    vs.retrieve()

    vs.video_sources.frames[0] = expected_original_left  # Hack for now, as colour space is changing !?!?
    vs.video_sources.frames[1] = expected_original_right  # Hack for now, as colour space is changing !?!?

    left, right = vs.get_undistorted()

    assert are_similar(left, expected_undistorted_left,
            threshold = 0.995, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.005)
    assert are_similar(right, expected_undistorted_right,
            threshold = 0.995, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.005)

    vs.set_extrinsic_parameters(r, t,
                                (int(vs.video_sources.frames[0].shape[1]),
                                 int(vs.video_sources.frames[0].shape[0] * 2)
                                 )
                                )  # double height
    rectified_left, rectified_right = vs.get_rectified()

    assert are_similar(rectified_left, expected_rectified_left,
            threshold = 0.995, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.005)
    assert are_similar(rectified_right, expected_rectified_right,
            threshold = 0.995, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.005)

