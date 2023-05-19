import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pixel_adjuster_pt import PixelAdjuster
from feature_extractor_pt import FeatureExtractor, FeatureMatcher
from pose_estimiation_pt import PoseEstimator
from bundle_adjuster_pt import BundleAdjuster

import kornia as K

from utils import get_pinhole_intrinsic_params, draw_matches, visualize_reprojection, compute_reprojection_error, get_noise_rotation
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
    # import ipdb; ipdb.set_trace()
    kp1, des1 = feature_extractor.extract(img1)
    kp2, des2 = feature_extractor.extract(img2)
    
    # scores, matches = K.feature.match_snn(des1, des2, 0.9)
    scores, matches = feature_matcher.match(des1, des2, kp1, kp2)
    # scores, matches = K.feature.match_fginn(des1, des2, kp1, kp2, mutual=True)

    # Now RANSAC
    src_pts = kp1[0, matches[:,0], :, 2].float().detach().requires_grad_(True)
    dst_pts = kp2[0, matches[:,1], :, 2].float().detach().requires_grad_(True)

    pixel_opt = PixelAdjuster(src_pts=src_pts, dst_pts=dst_pts, K=mtx_torch)

    num_iters = 2000

    start_time = time.time()

    err_normal_all = []

    for i in range(num_iters):
        # train
        reproj_2d_1_normal, reproj_2d_2_normal, err_normal = pixel_opt.adjust_step(pose_estimator)

        # check
        # make_dot(err, params={'R': R_opt, 'T': T_opt, 'point3d': point3d_opt}).render("err_torchviz", format="png")

        # print reprojection error
        print('==================== {}th Epoch ===================='.format(i))
        print('Normal average reprojection error: {}'.format(err_normal))

        err_normal_all.append(err_normal.item())

    # visualize projection
    visualize_reprojection(img1, img2, src_pts, dst_pts, reproj_2d_1_normal, reproj_2d_2_normal, key='normal')
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", execution_time, " seconds")

    # Create the plot
    plt.plot(err_normal_all, label='Normal')

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Reprojection Error')
    plt.title('Reprojection Error vs Epochs')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

    # point3d = point3d.detach().numpy()
    # # Visualize 3D points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(point3d[:, 0], point3d[:, 1], point3d[:, 2], c='b', marker='o')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.savefig('output/3d_points_opt.png')

    # fig2 = plt.figure()
    # new_img = draw_matches(img1, kp1[0, :, :, 2].data.cpu().numpy(), img2, kp2[0, :, :, 2].data.cpu().numpy(), matches.data.cpu().numpy(), inliers)
    # plt.imshow(new_img)
    # plt.savefig('output/matches.png')
    # plt.show()


if __name__ == '__main__':
    main()