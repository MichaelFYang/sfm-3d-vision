import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from feature_extractor_pt import FeatureExtractor, FeatureMatcher
from pose_estimiation_pt import PoseEstimator
from bundle_adjuster_pt import BundleAdjuster

import kornia as K

from utils import get_pinhole_intrinsic_params, draw_matches, visualize_reprojection, compute_reprojection_error, draw_camera_pose
import os
import argparse

# from torchviz import make_dot, make_dot_from_trace
import time
from tqdm import tqdm

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    return parser.parse_args()

def main():
    # set the seed to make each run deterministic
    torch.manual_seed(42)
    flags = read_args()
    dataset_name = flags.dataset

    curr_dir_path = os.getcwd()
    images_dir = os.path.join(curr_dir_path, 'dataset', dataset_name, 'rgb') 
    calibration_file_dir = os.path.join(curr_dir_path, 'dataset', dataset_name) 

    images_name = os.listdir(images_dir)

    # sort images by timestamp
    images_name = sorted(images_name, key=lambda x: float(x[:-4]))
    
    # read K from calibration file
    mtx = get_pinhole_intrinsic_params(calibration_file_dir)
    mtx_torch = torch.tensor(mtx).float()
    dist = np.zeros((5,))

    # Initialize feature extractor and feature matcher
    feature_extractor = FeatureExtractor(mtx, dist, method='sift')
    feature_matcher = FeatureMatcher()
    pose_estimator = PoseEstimator(mtx, dist)

    # Extract features and descriptors
    '''
    Keypoints correspond to specific locations in the image. cv2.KeyPoint
    Each keypoint is represented by a 2D coordinate (x, y) and a scale and orientation.

    Descriptors are vectors that describe the local appearance of the region around each keypoint.
    numpy array of size (num_keypoints x descriptor_size)
    '''
    R_t_0 = torch.tensor([[1,0,0,0], [0,1,0,0], [0,0,1,0]]).float()
    R_t_1 = torch.empty((3,4))
    P1 =  mtx_torch @ R_t_0
    P2 = torch.empty((3,4))
    point_3d_all = []

    pbar = tqdm(enumerate(images_name[:20:2]), total=len(images_name[:20:2]))
    # wrap tqdm around the loop to display progress bar
    for i, image_name in pbar:
        img_path = os.path.join(images_dir, image_name)
        img = cv2.imread(img_path)
        kp, des = feature_extractor.extract(img)

        if i == 0:
            prev_img = img
            prev_kp = kp
            prev_des = des
        else:
            scores, matches = feature_matcher.match(prev_des, des, prev_kp, kp)
            src_pts = prev_kp[0, matches[:,0], :, 2].float()
            dst_pts = kp[0, matches[:,1], :, 2].float()

            Fm, inliers = pose_estimator.compute_fundametal_matrix_kornia(src_pts, dst_pts)
            src_pts = src_pts[inliers]
            dst_pts = dst_pts[inliers]

            Em = K.geometry.essential_from_fundamental(Fm, mtx_torch, mtx_torch)

            R, T, _ = pose_estimator.recover_pose(Em, src_pts, dst_pts, mtx_torch)

            R_t_1[:3,:3] = R @ R_t_0[:3,:3]
            R_t_1[:3, 3] = R_t_0[:3, 3] + R_t_0[:3,:3] @ T.ravel()
            P2 = mtx_torch @ R_t_1

            points3d = pose_estimator.triangulate_points(P1, P2, src_pts, dst_pts)
            point_3d_all.append(points3d)

            # compute reprojection error and visualize to debug
            # reproj_2d_1, reproj_2d_2, err = compute_reprojection_error(points3d[:,:3], src_pts, dst_pts, P1=P1, P2=P2)
            # pbar.set_postfix({'Reprojection Error': err})

            # # print reprojection error
            # print('Average reprojection error: {}'.format(err))

            R_t_0 = R_t_1 
            P1 = P2

            prev_img = img
            prev_kp = kp
            prev_des = des

    point3d = torch.cat(point_3d_all, dim=0)
    # torch.save(point3d, 'point3d_hose_loop_50.pt')

    point3d = point3d.detach().numpy()

    # Visualize 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point3d[:, 0], point3d[:, 1], point3d[:, 2], s=1, cmap='gray') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    plt.savefig('output/3d_points_house_loop.png')


if __name__ == '__main__':
    main()