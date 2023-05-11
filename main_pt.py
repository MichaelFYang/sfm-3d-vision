import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from feature_extractor_pt import FeatureExtractor, FeatureMatcher
from pose_estimiation_pt import PoseEstimator

import kornia as K

from utils import get_pinhole_intrinsic_params, draw_matches
import os
import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    return parser.parse_args()

def main():
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
    # import ipdb; ipdb.set_trace()
    kp1, des1 = feature_extractor.extract(img1)
    kp2, des2 = feature_extractor.extract(img2)
    
    # scores, matches = K.feature.match_snn(des1, des2, 0.9)
    scores, matches = feature_matcher.match(des1, des2, kp1, kp2)
    # scores, matches = K.feature.match_fginn(des1, des2, kp1, kp2, mutual=True)

    # Now RANSAC
    src_pts = kp1[0, matches[:,0], :, 2].float()
    dst_pts = kp2[0, matches[:,1], :, 2].float()

    Fm, inliers = pose_estimator.compute_fundametal_matrix_kornia(src_pts, dst_pts)
    src_pts = src_pts[inliers]
    dst_pts = dst_pts[inliers]

    Em = K.geometry.essential_from_fundamental(Fm, mtx_torch, mtx_torch)

    R, T, point3d = pose_estimator.recover_pose(Em, src_pts, dst_pts, mtx_torch)

    new_img = draw_matches(img1, kp1[0, :, :, 2].data.cpu().numpy(), img2, kp2[0, :, :, 2].data.cpu().numpy(), matches.data.cpu().numpy(), inliers)
    
    plt.imshow(new_img)
    plt.savefig('output/matches.png')
    # plt.show()


if __name__ == '__main__':
    main()