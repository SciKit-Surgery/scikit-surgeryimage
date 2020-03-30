
import logging
from typing import Tuple
import numpy as np
import cv2

class PinholeCamera:
    """ Simple class to hold pinhole camera intrinsics. """
    def __init__(self, fx: float, fy: float, cx: float, cy: float, baseline: float = 1):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline
        
def undistort_and_rectify(l_img: np.ndarray, r_img: np.ndarray,
                          M1: np.ndarray, D1: np.ndarray, M2: np.ndarray,
                          D2: np.ndarray, R: np.ndarray, T: np.ndarray,
                          crop_to_roi: bool = True):
    """ Wrapper around openCV stereo rectificaiton functions,
     to make it slightly simpler to call. """

    if not np.array_equal(l_img.shape, r_img.shape):
        raise ValueError('Left and right images have different dimensions')

    w, h = l_img.shape[1], l_img.shape[0]
    R1 = np.zeros(shape=(3, 3))
    R2 = np.zeros(shape=(3, 3))
    P1 = np.zeros(shape=(3, 4))
    P2 = np.zeros(shape=(3, 4))
    Q = np.zeros(shape=(4, 4))

    logging.info("Rectifying/undistorting images")

    rois = cv2.stereoRectify(M1, D1, M2, D2, (w, h), R, T, R1, R2, P1, P2, Q,
                         flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    roi_l, roi_r = rois[-2:]

    undistort_rectify_map_l_x, undistort_rectify_map_l_y = \
        cv2.initUndistortRectifyMap(M1, D1, R1, P1, (w, h), cv2.CV_32FC1)

    undistort_rectify_map_r_x, undistort_rectify_map_r_y = \
        cv2.initUndistortRectifyMap(M2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    l_rect_image = cv2.remap(l_img, undistort_rectify_map_l_x,
                             undistort_rectify_map_l_y, cv2.INTER_LINEAR)

    r_rect_image = cv2.remap(r_img, undistort_rectify_map_r_x,
                             undistort_rectify_map_r_y, cv2.INTER_LINEAR)

    logging.debug(f'M1: {M1}')
    logging.debug(f'P1: {P1}')
    logging.debug(f'roi_l: {roi_l}')
    logging.debug(f'roi_r: {roi_r}')

    if crop_to_roi:
        # The roi for the left and right images is, in most cases, different.
        # We want the left and right images to be the same size, so crop the
        # images. The width/height needs to be the minimum of the two rois,
        # otherwise we will get some non-roi bits in one image.
        lx, ly, lw, lh = roi_l
        rx, ry, rw, rh = roi_r

        minh = np.min((lh, rh))
        minw = np.min((lw, rw))

        l_rect_image = l_rect_image[ly:ly+minh, lx:lx+minw,:]
        r_rect_image = r_rect_image[ry:ry+minh, rx:rx+minw,:]

    return l_rect_image, r_rect_image, P1, P2, roi_l, roi_r

def write_ply(ply_data: list, ply_file: str):
    """ Write data in ply format. """
    #TODO: Write extra properties?
    file = open(ply_file,"w")
    file.write('''ply 
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar nx
property uchar ny
property uchar nz
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(ply_data),"".join(ply_data)))

    file.close()

def write_pointcloud(points: np.ndarray, colours: np.ndarray, output_file: str):
    """ Write coloured point cloud data to file. """
    ply_data = []
    alpha = 0
    for j in range(len(points)):
        ply_data.append("%f %f %f %d %d %d %d %d %d %f\n"% 
                        (points[j][0], points[j][1], points[j][2],
                         1,1,1,
                        colours[j][0], colours[j][1], colours[j][2],
                        alpha))

    write_ply(ply_data, output_file)

def get_pointcloud(rgb: np.ndarray, disparity: np.ndarray, cam: PinholeCamera) \
     -> Tuple[np.ndarray, np.ndarray]:
     """ Generate a point cloud from rgb image, disparity map and camera
     intrinsics. """

    if not np.array_equal(rgb.shape[:2], disparity.shape[:2]):
        raise ValueError('RGB and Disparity images are different sizes')

    n_points = rgb.shape[0] * rgb.shape[1]
    colours = rgb.reshape((n_points, 3))

    cols = rgb.shape[0]
    rows = rgb.shape[1]

    Z = cam.fx * cam.baseline / disparity
    print(np.min(disparity))
    print(np.max(disparity))
    print(np.mean(disparity))

    x_ran = np.repeat(np.arange(cols).reshape((cols, 1)), rows, axis=1 )
    y_ran = np.repeat(np.arange(rows).reshape((1, rows)), cols)

    X = (x_ran - cam.cx) * Z / cam.fx
    Y = (np.arange(rgb.shape[1]) - cam.cy) * Z / cam.fy
    logging.debug(f'X range {np.max(X) - np.min(X)}')
    logging.debug(f'Y range {np.max(Y) - np.min(Y)}')
    logging.debug(f'Z range {np.max(Z) - np.min(Z)}')

    points = np.vstack((X.flatten(),Y.flatten(),Z.flatten()))
    return points.T, colours