import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from feature_extractor_pt import FeatureExtractor, FeatureMatcher
from pose_estimiation_pt import PoseEstimator
from bundle_adjuster_pt import BundleAdjuster

import kornia as K

from utils import get_pinhole_intrinsic_params, draw_matches, visualize_reprojection, compute_reprojection_error, visualize_pair_tensor_grayscale_images
import os
import argparse

# from torchviz import make_dot, make_dot_from_trace
import time

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

    interval = 10
    img1_path = os.path.join(images_dir, images_name[0])
    img1 = cv2.imread(img1_path)
    img2_path = os.path.join(images_dir, images_name[interval])
    img2 = cv2.imread(img2_path)

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
    img1_torch = K.image_to_tensor(img1, keepdim=False).float()
    img2_torch = K.image_to_tensor(img2, keepdim=False).float()

    img1_torch = K.color.rgb_to_grayscale(img1_torch).clone().detach().requires_grad_(True)
    img2_torch = K.color.rgb_to_grayscale(img2_torch).clone().detach().requires_grad_(True)
    
    num_iters = 20

    for _ in range(num_iters):
        # import ipdb; ipdb.set_trace()
        kp1, des1 = feature_extractor.extract(img1_torch)
        kp2, des2 = feature_extractor.extract(img2_torch)
        
        # scores, matches = K.feature.match_snn(des1, des2, 0.9)
        scores, matches = feature_matcher.match(des1, des2, kp1, kp2)
        # scores, matches = K.feature.match_fginn(des1, des2, kp1, kp2, mutual=True)

        # Now RANSAC
        src_pts = kp1[0, matches[:,0], :, 2].float()
        dst_pts = kp2[0, matches[:,1], :, 2].float()

        # with RANSAC (unstable gradient)
        # Fm, inliers = pose_estimator.compute_fundametal_matrix_kornia(src_pts, dst_pts)

        # without RANSAC
        # Fm = pose_estimator.compute_fundamental_matrix(src_pts, dst_pts)
        Fm = K.geometry.epipolar.find_fundamental(src_pts.unsqueeze(0), dst_pts.unsqueeze(0))

        # src_pts = src_pts[inliers]
        # dst_pts = dst_pts[inliers]

        Em = K.geometry.essential_from_fundamental(Fm.squeeze(), mtx_torch, mtx_torch)

        R, T, point3d = pose_estimator.recover_pose(Em, src_pts, dst_pts, mtx_torch)
        # R, T, point3d = K.geometry.epipolar.motion_from_essential_choose_solution(Em, mtx_torch, mtx_torch, src_pts, dst_pts, mask=None)

        reproj_2d_1, reproj_2d_2, err = compute_reprojection_error(point3d, src_pts, dst_pts, R=R, T=T, K=mtx_torch)
        
        # print reprojection error
        print('Average reprojection error: {}'.format(err))

        optimizer = torch.optim.Adam([img1_torch, img2_torch], lr=1)

        # clear gradients for this training step
        optimizer.zero_grad()

        # update R, T, point3d
        err.backward()
        optimizer.step()

        visualize_pair_tensor_grayscale_images(img1_torch, img2_torch)

if __name__ == '__main__':
    main()