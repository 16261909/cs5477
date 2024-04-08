""" CS4277/CS5477 Lab 4: Plane Sweep Stereo
See accompanying Jupyter notebook (lab4.ipynb) for instructions.

Name: Zhang Rongqi
Email: e1132299@u.nus.edu
NUSNET ID: e1132299

"""
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import scipy.ndimage

"""Helper functions: You should not have to touch the following functions.
"""
class Image(object):
    """
    Image class. You might find the following member variables useful:
    - image: RGB image (HxWx3) of dtype np.float64
    - pose_mat: 3x4 Camera extrinsics that transforms points from world to
        camera frame
    """
    def __init__(self, qvec, tvec, name, root_folder=''):
        self.qvec = qvec
        self.tvec = tvec
        self.name = name  # image filename
        self._image = self.load_image(os.path.join(root_folder, name))

        # Extrinsic matrix: Transforms from world to camera frame
        self.pose_mat = self.make_extrinsic(qvec, tvec)

    def __repr__(self):
        return '{}: qvec={}\n tvec={}'.format(
            self.name, self.qvec, self.tvec
        )

    @property
    def image(self):
        return self._image.copy()

    @staticmethod
    def load_image(path):
        """Loads image and converts it to float64"""
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im.astype(np.float64) / 255.0

    @staticmethod
    def make_extrinsic(qvec, tvec):
        """ Make 3x4 camera extrinsic matrix from colmap pose

        Args:
            qvec: Quaternion as per colmap format (q_cv) in the order
                  q_w, q_x, q_y, q_z
            tvec: translation as per colmap format (t_cv)

        Returns:

        """
        rotation = Rotation.from_quat(np.roll(qvec, -1))
        return np.concatenate([rotation.as_matrix(), tvec[:, None]], axis=1)

def write_json(outfile, images, intrinsic_matrix, img_hw):
    """Write metadata to json file.

    Args:
        outfile (str): File to write to
        images (list): List of Images
        intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix
        img_hw (tuple): (image height, image width)
    """

    img_height, img_width = img_hw

    images_meta = []
    for im in images:
        images_meta.append({
            'name': im.name,
            'qvec': im.qvec.tolist(),
            'tvec': im.tvec.tolist(),
        })

    data = {
        'img_height': img_height,
        'img_width': img_width,
        'K': intrinsic_matrix.tolist(),
        'images': images_meta
    }
    with open(outfile, 'w') as fid:
        json.dump(data, fid, indent=2)


def load_data(root_folder):
    """Loads dataset.

    Args:
        root_folder (str): Path to data folder. Should contain metadata.json

    Returns:
        images, K, img_hw
    """
    print('Loading data from {}...'.format(root_folder))
    with open(os.path.join(root_folder, 'metadata.json')) as fid:
        metadata = json.load(fid)

    images = []
    for im in metadata['images']:
        images.append(Image(np.array(im['qvec']), np.array(im['tvec']),
                            im['name'], root_folder=root_folder))
    img_hw = (metadata['img_height'], metadata['img_width'])
    K = np.array(metadata['K'])

    print('Loaded data containing {} images.'.format(len(images)))
    return images, K, img_hw


def invert_extrinsic(cam_matrix):
    """Invert extrinsic matrix"""
    irot_mat = cam_matrix[:3, :3].transpose()
    trans_vec = cam_matrix[:3, 3, None]

    inverted = np.concatenate([irot_mat,  -irot_mat @ trans_vec], axis=1)
    return inverted


def concat_extrinsic_matrix(mat1, mat2):
    """Concatenate two 3x4 extrinsic matrices, i.e. result = mat1 @ mat2
      (ignoring matrix dimensions)
    """
    r1, t1 = mat1[:3, :3], mat1[:3, 3:]
    r2, t2 = mat2[:3, :3], mat2[:3, 3:]
    rot = r1 @ r2
    trans = r1@t2 + t1
    concatenated = np.concatenate([rot, trans], axis=1)
    return concatenated


def rgb2hex(rgb):
    """Converts color representation into hexadecimal representation for K3D

    Args:
        rgb (np.ndarray): (N, 3) array holding colors

    Returns:
        hex (np.ndarray): array (N, ) of size N, each element indicates the
          color, e.g. 0x0000FF = blue
    """
    rgb_uint = (rgb * 255).astype(np.uint8)
    hex = np.sum(rgb_uint * np.array([[256 ** 2, 256, 1]]),
                 axis=1).astype(np.uint32)
    return hex

"""Functions to be implemented
"""
# Part 1
def get_plane_sweep_homographies(K, relative_pose, inv_depths):
    """Compute plane sweep homographies, assuming fronto parallel planes w.r.t.
    reference camera

    Args:
        K (np.ndarray): Camera intrinsic matrix (3,3)
        relative_pose (np.ndarray): Relative pose between the two cameras
          of shape (3, 4)
        inv_depths (np.ndarray): Inverse depths to warp of size (D, )

    Returns:
        homographies (D, 3, 3)
    """

    homographies = []

    """ YOUR CODE STARTS HERE """
    R_T = relative_pose[:, :3]
    C = R_T.T @ relative_pose[:, 3:] # why not negative? I'm confused.
    n_T = np.array([[0, 0, 1]]) # fronto-parallel plane
    K_inv = np.linalg.inv(K)
    M = R_T @ C @ n_T
    M = np.expand_dims(M, axis=0)
    M = np.repeat(M, inv_depths.shape[0], axis=0)
    M = np.einsum('dij,d->dij', M, inv_depths)
    homographies = np.einsum('ij,djk,kl->dil', K, R_T + M, K_inv)

    """ YOUR CODE ENDS HERE """

    return np.array(homographies)

# Part 2
def compute_plane_sweep_volume(images, ref_pose, K, inv_depths, img_hw):
    """Compute plane sweep volume, by warping all images to the reference camera
    fronto-parallel planes, before computing the variance for each pixel and
    depth.

    Args:
        images (list[Image]): List of images which contains information about
          the camera extrinsics for each image
        ref_pose (np.ndarray): Reference camera pose
        K (np.ndarray): 3x3 intrinsic matrix (assumed same for all cameras)
        inv_depths (list): List of inverse depths to consider for plane sweep
        img_hw (tuple): tuple containing (H, W), which are the output height
          and width for the plane sweep volume.

    Returns:
        ps_volume (np.ndarray):
          Plane sweep volume of size (D, H, W), with dtype=np.float64, where
          D is len(inv_depths), and (H, W) are the image heights and width
          respectively. Each element should contain the variance of all pixel
          intensities that warp onto it.
        accum_count (np.ndarray):
          Accumulator count of same size as ps_volume, and dtype=np.int32.
          Keeps track of how many images are warped into a certain pixel,
          i.e. the number of pixels used to compute the variance.
    """

    D = len(inv_depths)
    H, W = img_hw
    ps_volume = np.zeros((D, H, W), dtype=np.float64)
    accum_count = np.zeros((D, H, W), dtype=np.int32)

    """ YOUR CODE STARTS HERE """

    kernel_size = 3
    threshold = 1e-5
    for i in range(len(images)):
        if np.linalg.norm(images[i].pose_mat - ref_pose) < threshold:
            ref_id = i
            break
    assert(ref_id != 0)
    print(ref_id)

    for i in range(len(images)):
        print(f'Processing image {i}')
        relative_pose = concat_extrinsic_matrix(ref_pose, invert_extrinsic(images[i].pose_mat))
        homographies = get_plane_sweep_homographies(K, relative_pose, inv_depths)
        # relative_pose = concat_extrinsic_matrix(images[i].pose_mat, invert_extrinsic(ref_pose))
        # homographies[i] = get_plane_sweep_homographies(K, relative_pose, inv_depths)
        for d in range(D):
            warped_image = cv2.warpPerspective(images[i].image, homographies[d], (W, H))
            warped_mask = cv2.warpPerspective(np.ones((H, W), dtype=np.float64), homographies[d], (W, H))

            for dx in range(-kernel_size // 2, kernel_size // 2 + 1):
                for dy in range(-kernel_size // 2, kernel_size // 2 + 1):
                    x_indices = np.clip(np.arange(W) + dx, 0, W - 1)
                    y_indices = np.clip(np.arange(H) + dy, 0, H - 1)
                    valid_mask = warped_mask[y_indices][:, x_indices] != 0
                    ps_volume[d] += np.sum(np.abs(images[ref_id].image[y_indices][:, x_indices] - warped_image[y_indices][:, x_indices]) * valid_mask[..., None], axis=-1)
                    accum_count[d] += np.where(warped_mask[y_indices][:, x_indices] > 0, 1, 0)

    ps_volume /= accum_count

    # Following code is for compute 'variance'

    # homographies = np.zeros((len(images), D, 3, 3), dtype=np.float64)
    # ps_sum = np.zeros((D, H, W, 3), dtype=np.float64)
    # warped_images = np.zeros((len(images), D, H, W, 3), dtype=np.float64)
    # warped_maskes = np.zeros((len(images), D, H, W), dtype=np.float64)
    #
    # for i in range(len(images)):
    #     print(f'Processing image {i}')
    #     relative_pose = concat_extrinsic_matrix(ref_pose, invert_extrinsic(images[i].pose_mat))
    #     homographies[i] = get_plane_sweep_homographies(K, relative_pose, inv_depths)
    #     for d in range(D):
    #         warped_images[i, d] = cv2.warpPerspective(images[i].image, homographies[i, d], (W, H))
    #         warped_maskes[i, d] = cv2.warpPerspective(np.ones((H, W), dtype=np.float64), homographies[i, d], (W, H))
    #
    # ps_sum = np.sum(warped_images, axis=0)
    # accum_count[:] += np.sum(np.where(warped_maskes > 0, 1, 0), axis=0)
    #
    # ps_sum /= accum_count[..., None]
    #
    # for i in range(len(images)):
    #     # Shouldn't count unmapped pixels! Debugging it cost me one day!
    #     # cannot use vector operation here. It will cause OOM error.
    #     ps_volume[:] += np.sum(np.where(np.expand_dims(warped_maskes[i, :] > 0, axis=-1), warped_images[i, :] - ps_sum[:], 0) ** 2, axis=3)
    #
    # ps_volume /= accum_count * 3


    # def print1(x, y):
    #     filename = f'{x}_{y}.txt'
    #     with open(filename, 'w') as f:
    #         f.write("-------------------\n")
    #         f.write(f"ps_sum: {ps_sum[:, y, x]}\n")
    #         f.write(f"warped_images: {warped_images[:, 0, y, x]}\n")
    #         f.write(f"warped_images: {warped_images[:, 1, y, x]}\n")
    #         f.write(f"warped_maskes: {warped_maskes[:, 0, y, x]}\n")
    #         f.write(f"warped_maskes: {warped_maskes[:, 1, y, x]}\n")
    #
    #         f.write(f"accum_count: {accum_count[:, y, x]}\n")
    #         f.write(f"ps_volume: {ps_volume[:, y, x]}\n")
    #         f.write(f"min_d: {np.argmin(ps_volume[:, y, x])}\n")
    #         f.write(f"(warped_images[i, :, y, x] - ps_sum[:, y, x]): {(warped_images[i, :, y, x] - ps_sum[:, y, x])}\n")
    #         f.write("-------------------\n")
    #
    # print1(20, 200)
    # print1(300, 100)
    #
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 14))
    # for d in range(D):
    #     plt.subplot(D//2, 2, d + 1)
    #     plt.imshow(accum_count[d].astype(np.uint8) / np.max(accum_count[d]), cmap='gray')
    #     plt.title(f'{np.max(accum_count[d])}')

    """ YOUR CODE ENDS HERE """
    print(ps_volume.shape)

    return ps_volume, accum_count


def compute_depths(ps_volume, inv_depths):
    """Computes inverse depth map from plane sweep volume as the
    argmin over plane sweep volume variances.

    Args:
        ps_volume (np.ndarray): Plane sweep volume of size (D, H, W) from
          compute_plane_sweep_volume()
        inv_depths (np.ndarray): List of depths considered in the plane
          sweeping (D,)

    Returns:
        inv_depth_image (np.ndarray): inverse-depth estimate (H, W)
    """

    inv_depth_image = np.zeros(ps_volume.shape[1:], dtype=np.float64)

    """ YOUR CODE STARTS HERE """

    inv_depth_image = inv_depths[np.argmin(ps_volume, axis=0)]

    """ YOUR CODE ENDS HERE """

    return inv_depth_image


# Part 3
def post_process(ps_volume, inv_depths, accum_count):
    """Post processes the plane sweep volume and compute a mask to indicate
    which pixels have confident estimates of the depth

    Args:
        ps_volume: Plane sweep volume from compute_plane_sweep_volume()
          of size (D, H, W)
        inv_depths (List[float]): List of depths considered in the plane
          sweeping
        accum_count: Accumulator count from compute_plane_sweep_volume(), which
          can be used to indicate which pixels are not observed by many other
          images.

    Returns:
        inv_depth_image: Denoised Inverse depth image (similar to compute_depths)
        mask: np.ndarray of size (H, W) and dtype np.bool.
          Pixels with values TRUE indicate valid pixels.
    """

    mask = np.ones(ps_volume.shape[1:], dtype=np.int32)
    inv_depth_image = np.zeros(ps_volume.shape[1:], dtype=np.float64)
    #print(accum_count)
    """ YOUR CODE STARTS HERE """


    smooth_ps_volume = scipy.ndimage.median_filter(ps_volume, size=(3, 5, 5))
    variance_threshold = 0.001
    accum_threshold = 0.5
    print(smooth_ps_volume.max(), smooth_ps_volume.min(), smooth_ps_volume.mean(), smooth_ps_volume.var())
    ps_mean = np.mean(smooth_ps_volume, axis=(1, 2))
    ps_var = np.var(smooth_ps_volume, axis=(1, 2))
    print(accum_count.max(), accum_count.min(), accum_count.mean(), accum_count.var())
    accum_mean = np.mean(accum_count, axis=(1, 2))
    accum_var = np.var(accum_count, axis=(1, 2))
    idx = np.argmin(smooth_ps_volume, axis=0)
    for h in range(ps_volume.shape[1]):
        for w in range(ps_volume.shape[2]):
            # cannot use vector operation here. It will cause OOM error.
            if accum_count[idx[h, w], h, w] < accum_mean[idx[h, w]] - accum_var[idx[h, w]] or smooth_ps_volume[idx[h, w], h, w] > ps_mean[idx[h, w]] + ps_var[idx[h, w]]:
                mask[h, w] = 0
    inv_depth_image = inv_depths[idx]

    """ YOUR CODE ENDS HERE """

    return inv_depth_image, mask


# Part 4
def unproject_depth_map(image, inv_depth_image, K, mask=None):
    """Converts the depth map into points by unprojecting depth map into 3D

    Note: You will also need to implement the case where no mask is provided

    Args:
        image (np.ndarray): Image bitmap (H, W, 3)
        inv_depth_image (np.ndarray): Inverse depth image (H, W)
        K (np.ndarray): 3x3 Camera intrinsics
        mask (np.ndarray): Optional mask of size (H, W) and dtype=np.bool.

    Returns:
        xyz (np.ndarray): Nx3 coordinates of points, dtype=np.float64.
        rgb (np.ndarray): Nx3 RGB colors, where rgb[i, :] is the (Red,Green,Blue)
          colors for the points at position xyz[i, :]. Should be in the range
          [0, 1] and have dtype=np.float64.
    """

    xyz = np.zeros([0, 3], dtype=np.float64)
    rgb = np.zeros([0, 3], dtype=np.float64)  # values should be within (0, 1)
    H, W = image.shape[0:2]
    """ YOUR CODE STARTS HERE """

    if mask is None:
        # my numpy version is 1.21.6, which does not support np.bool. So I have to use np.int32
        mask = np.ones_like(inv_depth_image, dtype=np.int32)
    mask = mask.flatten()
    mask_idx = np.where(mask > 0)
    z = 1 / inv_depth_image
    idx_x, idx_y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    idx_x = idx_x.flatten()[mask_idx]
    idx_y = idx_y.flatten()[mask_idx]
    z = z[idx_y, idx_x]
    print(z.shape, idx_x.shape)
    X = np.stack([z * idx_x, z * idx_y, z], axis=-1)
    points3d = (np.linalg.inv(K) @ X.T).T
    pointsrgb = image[idx_y, idx_x]

    """ YOUR CODE ENDS HERE """

    xyz = np.array(points3d)
    rgb = np.array(pointsrgb)
    return xyz, rgb
