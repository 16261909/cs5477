""" CS4277/CS5477 Lab 1: Metric Rectification and Robust Homography Estimation.
See accompanying file (lab1.pdf) for instructions.

Name: Zhang Rongqi
Email: e1132299@u.nus.edu
Student ID: A0276566M
"""

import numpy as np
import cv2
from helper import *
from math import floor, ceil, sqrt


def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """

    h_matrix = np.eye(3, dtype=np.float64)

    """ YOUR CODE STARTS HERE """
    # Compute normalization matrix
    centroid_src = np.mean(src, axis=0)
    d_src = np.linalg.norm(src - centroid_src[None, :], axis=1)
    s_src = sqrt(2) / np.mean(d_src)
    T_norm_src = np.array([[s_src, 0.0, -s_src * centroid_src[0]],
                           [0.0, s_src, -s_src * centroid_src[1]],
                           [0.0, 0.0, 1.0]])

    centroid_dst = np.mean(dst, axis=0)
    d_dst = np.linalg.norm(dst - centroid_dst[None, :], axis=1)
    s_dst = sqrt(2) / np.mean(d_dst)
    T_norm_dst = np.array([[s_dst, 0.0, -s_dst * centroid_dst[0]],
                           [0.0, s_dst, -s_dst * centroid_dst[1]],
                           [0.0, 0.0, 1.0]])

    srcn = transform_homography(src, T_norm_src)
    dstn = transform_homography(dst, T_norm_dst)

    # Compute homography
    n_corr = srcn.shape[0]
    A = np.zeros((n_corr * 2, 9), dtype=np.float64)
    for i in range(n_corr):
        A[2 * i, 0] = srcn[i, 0]
        A[2 * i, 1] = srcn[i, 1]
        A[2 * i, 2] = 1.0
        A[2 * i, 6] = -dstn[i, 0] * srcn[i, 0]
        A[2 * i, 7] = -dstn[i, 0] * srcn[i, 1]
        A[2 * i, 8] = -dstn[i, 0] * 1.0

        A[2 * i + 1, 3] = srcn[i, 0]
        A[2 * i + 1, 4] = srcn[i, 1]
        A[2 * i + 1, 5] = 1.0
        A[2 * i + 1, 6] = -dstn[i, 1] * srcn[i, 0]
        A[2 * i + 1, 7] = -dstn[i, 1] * srcn[i, 1]
        A[2 * i + 1, 8] = -dstn[i, 1] * 1.0

    u, s, vt = np.linalg.svd(A)
    h_matrix_n = np.reshape(vt[-1, :], (3, 3))

    # Unnormalize homography
    h_matrix = np.linalg.inv(T_norm_dst) @ h_matrix_n @ T_norm_src
    h_matrix /= h_matrix[2, 2]

    # src = src.astype(np.float32)
    # dst = dst.astype(np.float32)
    # h_matrix = cv2.findHomography(src, dst)[0].astype(np.float64)
    """ YOUR CODE ENDS HERE """

    return h_matrix


def transform_homography(src, h_matrix):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    Prohibited functions:
        cv2.perspectiveTransform()

    """
    transformed = None

    """ YOUR CODE STARTS HERE """

    homogeneous_src = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
    homogeneous_transformed = np.dot(h_matrix, homogeneous_src.T).T
    transformed = homogeneous_transformed[:, :2] / homogeneous_transformed[:, 2:3]

    """ YOUR CODE ENDS HERE """

    return transformed


def warp_image(src, dst, h_matrix):
    """Applies perspective transformation to source image to warp it onto the
    destination (background) image

    Args:
        src (np.ndarray): Source image to be warped
        dst (np.ndarray): Background image to warp template onto
        h_matrix (np.ndarray): Warps coordinates from src to the dst, i.e.
                                 x_{dst} = h_matrix * x_{src},
                               where x_{src}, x_{dst} are the homogeneous
                               coordinates in I_{src} and I_{dst} respectively

    Returns:
        dst (np.ndarray): Source image warped onto destination image

    Prohibited functions:
        cv2.warpPerspective()
    You may use the following functions: np.meshgrid(), cv2.remap(), transform_homography()
    """
    dst = dst.copy()  # deep copy to avoid overwriting the original image

    """ YOUR CODE STARTS HERE """
    h_dst, w_dst = dst.shape[0:2]
    h_matrix_inv = np.linalg.inv(h_matrix)
    y, x = np.meshgrid(np.arange(h_dst), np.arange(w_dst), indexing='ij')
    homogeneous_dst_coords = np.stack((x.flatten(), y.flatten()))
    dst_transformed_coords = transform_homography(homogeneous_dst_coords.T, h_matrix_inv)

    map_x = dst_transformed_coords[:, 0].reshape(h_dst, w_dst).astype(np.float32)
    map_y = dst_transformed_coords[:, 1].reshape(h_dst, w_dst).astype(np.float32)
    src_transformed = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
                                dst=dst)
    mask = np.any(src_transformed != [0, 0, 0], axis=-1)
    dst[mask] = src_transformed[mask]

    """ YOUR CODE ENDS HERE """
    # cv2.warpPerspective(src, h_matrix, dsize=dst.shape[1::-1],
    #                     dst=dst, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def compute_affine_rectification(src_img: np.ndarray, lines_vec: list):
    '''
       The first step of the stratification method for metric rectification. Compute
       the projective transformation matrix Hp with line at infinity. At least two
       parallel line pairs are required to obtain the vanishing line. Then warping
       the image with the predicted projective transformation Hp to recover the affine
       properties. X_dst=Hp*X_src

       Args:
           src_img: Original image X_src
           lines_vec: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Affinely rectified image by removing projective distortion

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    Hp = np.zeros((3, 3))
    """ YOUR CODE STARTS HERE """
    point0, point1 = lines_vec[0].intersetion_point(lines_vec[1]), lines_vec[2].intersetion_point(lines_vec[3])
    l_inf = Line_Equation(point0, point1)

    # print(point0.coordinate, point1.coordinate, l_inf)
    Hp = np.array([[1, 0, -l_inf[0] / l_inf[2]], [0, 1, -l_inf[1] / l_inf[2]], [0, 0, 1 / l_inf[2]]]).T

    Hp_prime = np.linalg.inv(Hp)

    def construct_hs(H, original_width, original_height):
        point00 = np.array([0, 0, 1])
        point10 = np.array([original_width - 1, 0, 1])
        point01 = np.array([0, original_height - 1, 1])
        point11 = np.array([original_width - 1, original_height - 1, 1])

        # print(point00, point01, point10, point11)

        point00_prime = H @ point00
        point01_prime = H @ point01
        point10_prime = H @ point10
        point11_prime = H @ point11

        point00_prime = point00_prime / point00_prime[2]
        point01_prime = point01_prime / point01_prime[2]
        point10_prime = point10_prime / point10_prime[2]
        point11_prime = point11_prime / point11_prime[2]

        x_max = max(point01_prime[0], point11_prime[0], point10_prime[0], point00_prime[0])
        x_min = min(point01_prime[0], point11_prime[0], point10_prime[0], point00_prime[0])
        y_max = max(point01_prime[1], point11_prime[1], point10_prime[1], point00_prime[1])
        y_min = min(point01_prime[1], point11_prime[1], point10_prime[1], point00_prime[1])

        scale_x = (x_max - x_min) / (original_width - 1)
        scale_y = (y_max - y_min) / (original_height - 1)
        scale = max(scale_x, scale_y)
        sx = 1 / scale * (1 if point10_prime[0] > 0 else -1)
        sy = 1 / scale * (1 if point01_prime[1] > 0 else -1)
        center_point = sx * (x_min + x_max) / 2, sy * (y_min + y_max) / 2
        tx = original_width / 2 - center_point[0]
        ty = original_height / 2 - center_point[1]

        Ha = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]])
        return Ha

    Hs_after = construct_hs(Hp_prime, dst.shape[1], dst.shape[0])

    dst = warp_image(src_img, dst, Hs_after @ Hp_prime)
    # dst = warp_image(src_img, dst, Hp_prime)
    """ YOUR CODE ENDS HERE """

    return dst


def compute_metric_rectification_step2(src_img: np.ndarray, line_vecs: list):
    '''
       The second step of the stratification method for metric rectification. Compute
       the affine transformation Ha with the degenerate conic from at least two
       orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=Ha*X_src

       Args:
           src_img: Affinely rectified image X_src
           line_vecs: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           X_dst: Image after metric rectification

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    Ha = np.zeros((3, 3))

    """ YOUR CODE STARTS HERE """
    A = np.stack([[line_vecs[0].vec_para[0] * line_vecs[1].vec_para[0],
                   line_vecs[0].vec_para[1] * line_vecs[1].vec_para[0] + line_vecs[0].vec_para[0] *
                   line_vecs[1].vec_para[1], line_vecs[0].vec_para[1] * line_vecs[1].vec_para[1]],
                  [line_vecs[2].vec_para[0] * line_vecs[3].vec_para[0],
                   line_vecs[2].vec_para[1] * line_vecs[3].vec_para[0] + line_vecs[2].vec_para[0] *
                   line_vecs[3].vec_para[1], line_vecs[2].vec_para[1] * line_vecs[3].vec_para[1]]])
    U, S, VT = np.linalg.svd(A)
    s = VT.T[:, -1]  # transpose of V
    KKT = np.array([[s[0], s[1]], [s[1], s[2]]])
    K = np.linalg.cholesky(KKT)
    Ha = np.array([[K[0, 0], K[0, 1], 0], [K[1, 0], K[1, 1], 0], [0, 0, 1]])
    inv_Ha = np.linalg.inv(Ha)

    # print(K, K.T, K @ K.T, KKT, sep='\n')

    def construct_hs(H, original_width, original_height):
        point00 = np.array([0, 0, 1])
        point10 = np.array([original_width - 1, 0, 1])
        point01 = np.array([0, original_height - 1, 1])
        point11 = np.array([original_width - 1, original_height - 1, 1])

        # print(point00, point01, point10, point11)

        point00_prime = H @ point00
        point01_prime = H @ point01
        point10_prime = H @ point10
        point11_prime = H @ point11

        point00_prime = point00_prime / point00_prime[2]
        point01_prime = point01_prime / point01_prime[2]
        point10_prime = point10_prime / point10_prime[2]
        point11_prime = point11_prime / point11_prime[2]

        x_max = max(point01_prime[0], point11_prime[0], point10_prime[0], point00_prime[0])
        x_min = min(point01_prime[0], point11_prime[0], point10_prime[0], point00_prime[0])
        y_max = max(point01_prime[1], point11_prime[1], point10_prime[1], point00_prime[1])
        y_min = min(point01_prime[1], point11_prime[1], point10_prime[1], point00_prime[1])

        scale_x = (x_max - x_min) / (original_width - 1)
        scale_y = (y_max - y_min) / (original_height - 1)
        scale = max(scale_x, scale_y)
        sx = 1 / scale * (1 if point10_prime[0] > 0 else -1)
        sy = 1 / scale * (1 if point01_prime[1] > 0 else -1)
        center_point = sx * (x_min + x_max) / 2, sy * (y_min + y_max) / 2
        tx = original_width / 2 - center_point[0]
        ty = original_height / 2 - center_point[1]

        Ha = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]])
        return Ha

    Hs_after = construct_hs(inv_Ha, dst.shape[1], dst.shape[0])
    dst = warp_image(src_img, dst, Hs_after @ inv_Ha)
    """ YOUR CODE ENDS HERE """

    return dst


def compute_metric_rectification_one_step(src_img: np.ndarray, line_vecs: list):
    '''
       One-step metric rectification. Compute the transformation matrix H (i.e. H=HaHp) directly
       from five orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=H*X_src
       Args:
           src_img: Original image Xc
           line_infinity: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Image after metric rectification

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    H = np.zeros((3, 3))

    """ YOUR CODE STARTS HERE """
    A = np.zeros((5, 6))
    for i in range(5):
        a, b, c = line_vecs[2 * i].vec_para
        d, e, f = line_vecs[2 * i + 1].vec_para
        A[i] = [a * d, (b * d + a * e) / 2, b * e, (a * f + c * d) / 2, (b * f + c * e) / 2, f * c]

    U, S, VT = np.linalg.svd(A)
    s = VT.T[:, -1]
    C_inf_star_prime = np.array([[s[0], s[1] / 2, s[3] / 2], [s[1] / 2, s[2], s[4] / 2], [s[3] / 2, s[4] / 2, s[5]]])
    U, S, VT = np.linalg.svd(C_inf_star_prime)
    S[2] = 0
    S[1] = S[0]
    C_inf_star_prime_approx = U @ np.diag(S) @ VT

    H = U
    inv_H = np.linalg.inv(H)

    def construct_hs(H, original_width, original_height):
        point00 = np.array([0, 0, 1])
        point10 = np.array([original_width - 1, 0, 1])
        point01 = np.array([0, original_height - 1, 1])
        point11 = np.array([original_width - 1, original_height - 1, 1])

        # print(point00, point01, point10, point11)

        point00_prime = H @ point00
        point01_prime = H @ point01
        point10_prime = H @ point10
        point11_prime = H @ point11

        point00_prime = point00_prime / point00_prime[2]
        point01_prime = point01_prime / point01_prime[2]
        point10_prime = point10_prime / point10_prime[2]
        point11_prime = point11_prime / point11_prime[2]

        x_max = max(point01_prime[0], point11_prime[0], point10_prime[0], point00_prime[0])
        x_min = min(point01_prime[0], point11_prime[0], point10_prime[0], point00_prime[0])
        y_max = max(point01_prime[1], point11_prime[1], point10_prime[1], point00_prime[1])
        y_min = min(point01_prime[1], point11_prime[1], point10_prime[1], point00_prime[1])

        scale_x = (x_max - x_min) / (original_width - 1)
        scale_y = (y_max - y_min) / (original_height - 1)
        scale = max(scale_x, scale_y)
        sx = 1 / scale * (1 if point10_prime[0] > 0 else -1)
        sy = 1 / scale * (1 if point01_prime[1] > 0 else -1)
        center_point = sx * (x_min + x_max) / 2, sy * (y_min + y_max) / 2
        tx = original_width / 2 - center_point[0]
        ty = original_height / 2 - center_point[1]

        Ha = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]])
        return Ha

    Hs_after = construct_hs(inv_H, dst.shape[1], dst.shape[0])
    # print(np.linalg.det(inv_H), np.linalg.det(Hs_after))
    #
    # src_img[0, :, :] = 255
    # src_img[1, :, :] = 255
    # src_img[-1, :, :] = 255
    # src_img[:, 0, :] = 255
    # src_img[:, 1, :] = 255
    # src_img[:, -1, :] = 255
    #
    # print(H)

    dst = warp_image(src_img, dst, Hs_after @ inv_H)
    """ YOUR CODE ENDS HERE """

    return dst


def compute_homography_error(src, dst, homography):
    """Compute the squared bidirectional pixel reprojection error for
    provided correspondences

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        homography (np.ndarray): Homography matrix that transforms src to dst.

    Returns:
        err (np.ndarray): Array of size (N, ) containing the error d for each
        correspondence, computed as:
          d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
        where ||a|| denotes the l2 norm (euclidean distance) of vector a.
    """
    d = np.zeros(src.shape[0], np.float64)

    """ YOUR CODE STARTS HERE """
    src_prime = transform_homography(src, homography)
    d = d + np.linalg.norm(dst - src_prime, axis=1) ** 2
    dst_prime = transform_homography(dst, np.linalg.inv(homography))
    d = d + np.linalg.norm(src - dst_prime, axis=1) ** 2

    """ YOUR CODE ENDS HERE """

    return d


def compute_homography_ransac(src, dst, thresh=16.0, num_tries=200):
    """Calculates the perspective transform from at least 4 points of
    corresponding points in a robust manner using RANSAC. After RANSAC, all the
    inlier correspondences will be used to re-estimate the homography matrix.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        thresh (float): Maximum allowed squared bidirectional pixel reprojection
          error to treat a point pair as an inlier (default: 16.0). Pixel
          reprojection error is computed as:
            d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
          where ||a|| denotes the l2 norm (euclidean distance) of vector a.
        num_tries (int): Number of trials for RANSAC

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.
        mask (np.ndarraay): Output mask with dtype np.bool where 1 indicates
          inliers

    Prohibited functions:
        cv2.findHomography()
    """

    h_matrix = np.eye(3, dtype=np.float64)
    mask = np.ones(src.shape[0], dtype=np.bool)

    """ YOUR CODE STARTS HERE """
    max_inliners = 0
    eps = 1e-5

    def collinear(p1, p2, p3):
        return abs(np.cross(p2 - p1, p3 - p1)) < eps

    for i in range(num_tries):
        while True:
            indices = np.random.choice(src.shape[0], 4, replace=True)
            if collinear(src[0], src[1], src[2]) or collinear(src[0], src[1], src[3]) or collinear(src[0], src[2], src[3]) or collinear(src[1], src[2], src[3]):
                continue
            else:
                break

        H = compute_homography(src[indices], dst[indices])
        d = compute_homography_error(src, dst, H)
        mask_new = d < thresh
        inliners = np.sum(mask_new)
        if inliners > max_inliners:
            max_inliners = inliners
            mask = mask_new
            h_matrix = H
    """ YOUR CODE ENDS HERE """

    return h_matrix, mask
