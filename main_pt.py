import cv2
import numpy as np
import matplotlib.pyplot as plt
from feature_extractor_pt import FeatureExtractor, FeatureMatcher

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
    dist = np.zeros((5,))

    interval = 10
    img1_path = os.path.join(images_dir, images_name[0])
    img1 = cv2.imread(img1_path)
    img2_path = os.path.join(images_dir, images_name[interval])
    img2 = cv2.imread(img2_path)

    # Initialize feature extractor and feature matcher
    feature_extractor = FeatureExtractor(mtx, dist, method='sift')
    feature_matcher = FeatureMatcher(matcher='flann')

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
    scores, matches = K.feature.match_fginn(des1, des2, kp1, kp2, mutual=True)

    # Now RANSAC
    src_pts = kp1[0, matches[:,0], :, 2].data.cpu().numpy()
    dst_pts = kp2[0, matches[:,1], :, 2].data.cpu().numpy()

    Fm, inliers = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)

    new_img = draw_matches(img1, kp1[0, :, :, 2].data.cpu().numpy(), img2, kp2[0, :, :, 2].data.cpu().numpy(), matches.data.cpu().numpy(), inliers)
    
    plt.imshow(new_img)
    plt.show()


if __name__ == '__main__':
    main()