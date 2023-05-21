import cv2
import torch
import kornia as K
import numpy as np
import matplotlib.pyplot as plt
from feature_extractor_pt import FeatureExtractor, FeatureMatcher
from pose_estimiation_pt import PoseEstimator

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
    # torch.manual_seed(42)
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
    camera_pose_all = [R_t_0[:3,3]]

    # Visualize 3D points
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # Iterate through frames with interval
    # Initialize map from 2D keypoints to 3D points
    map_2d_to_3d = {}

    interval = 1
    end = 3
    pbar = tqdm(enumerate(images_name[:end:interval]), total=len(images_name[:end:interval]))
    # wrap tqdm around the loop to display progress bar
    for i, image_name in pbar:
        img_path = os.path.join(images_dir, image_name)
        img = cv2.imread(img_path)
        kp, des = feature_extractor.extract(img)

        if i == 0:
            prev_kp = kp
            prev_des = des
        elif i == 1:
            _, matches = feature_matcher.match(prev_des, des, prev_kp, kp)
            src_pts = prev_kp[0, matches[:,0], :, 2].float()
            dst_pts = kp[0, matches[:,1], :, 2].float()

            Fm, inliers = pose_estimator.compute_fundametal_matrix_kornia(src_pts, dst_pts)
            src_pts = src_pts[inliers]
            dst_pts = dst_pts[inliers]

            Em = K.geometry.essential_from_fundamental(Fm, mtx_torch, mtx_torch)

            R, T, _ = pose_estimator.recover_pose(Em, src_pts, dst_pts, mtx_torch)

            R_t_1[:3,:3] = R @ R_t_0[:3,:3]
            R_t_1[:3, 3] = R_t_0[:3, 3] + R_t_0[:3,:3] @ T.ravel()

            camera_pose_all.append(R_t_1[:3, 3].data.clone())

            P2 = mtx_torch @ R_t_1
            points3d = pose_estimator.triangulate_points(P1, P2, src_pts, dst_pts)
            point_3d_all.append(points3d.data.clone())

            map_2d_to_3d = {tuple(pt2d.tolist()): pt3d for pt2d, pt3d in zip(dst_pts, points3d)}
            P1 = P2.clone()
            
            prev_kp = kp
            prev_des = des
        else:
            _, matches = feature_matcher.match(prev_des, des, prev_kp, kp)
            src_pts = prev_kp[0, matches[:,0], :, 2].float()
            dst_pts = kp[0, matches[:,1], :, 2].float()
            
            
            # Instead of estimating Fundamental matrix and Essential matrix,
            # check if there are enough 2D-3D correspondences in map_2d_to_3d
            src_pts_3d = [map_2d_to_3d.get(tuple(p.tolist()), None) for p in src_pts]
            
            if src_pts_3d.count(None) < len(src_pts_3d) - 8:
                # There are enough 2D-3D correspondences, use PnP to estimate pose
                valid = [p is not None for p in src_pts_3d]
                dst_pts_2d = dst_pts[valid]
                src_pts_3d = torch.cat([p.unsqueeze(dim=0) for p in src_pts_3d if p is not None])

                R_t_1 = K.geometry.solve_pnp_dlt(src_pts_3d[:, :-1].unsqueeze(dim=0), dst_pts_2d.unsqueeze(dim=0), mtx_torch.unsqueeze(dim=0))[0]
                
                # src_pts_3d_np = src_pts_3d.detach().numpy()
                # src_pts_2d_np = src_pts_2d.detach().numpy()
                # _, R_vec, T = cv2.solvePnP(src_pts_3d_np[:, :-1], src_pts_2d_np, mtx, np.zeros((4,1)))
                # R, _ = cv2.Rodrigues(R_vec)

                P2 = mtx_torch @ R_t_1
                
                # update camera poses
                camera_pose_all.append(R_t_1[:3, 3].data.clone())
                
                # Try to triangulate points. If PnP was used, this might fail for some points,
                # as the new 2D points might not correspond to the same 3D points as before.
                # But that's ok, as long as some points can be triangulated, they can be added to the map.
                try:
                    _, inliers = pose_estimator.compute_fundametal_matrix_kornia(src_pts, dst_pts)
                    src_pts = src_pts[inliers]
                    dst_pts = dst_pts[inliers]
                    
                    points3d = pose_estimator.triangulate_points(P1, P2, src_pts, dst_pts)
                    point_3d_all.append(points3d.data.clone())

                    # Update map_2d_to_3d with newly triangulated points
                    map_2d_to_3d = {tuple(pt2d.tolist()): pt3d for pt2d, pt3d in zip(dst_pts, points3d)}

                except:
                    pass

                R_t_0 = R_t_1 
                P1 = P2.clone()

                prev_kp = kp
                prev_des = des
            else:
                print('Not enough 2D-3D correspondences, skipping frame')
                
    # point_3d_all = torch.cat(point_3d_all, dim=0)
    point_3d_all = point_3d_all[1]
    # torch.save(point3d, 'point3d_hose_loop_50.pt')

    camera_pose_all = torch.stack(camera_pose_all, dim=0)

    point_3d_all = point_3d_all.detach().numpy()
    camera_pose_all = camera_pose_all.detach().numpy()

    # Visualize 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_3d_all[:, 0], point_3d_all[:, 1], point_3d_all[:, 2], s=5, cmap='blue') 
    
    # Plot the camera position as a point
    ax.scatter(camera_pose_all[:, 0], camera_pose_all[:, 1], camera_pose_all[:, 2], s=10, color='r')

    # Adjust plot limits
    # ax.set_xlim([-50, 50])  # Set appropriate values for xmin and xmax
    # ax.set_ylim([-50, 50])  # Set appropriate values for ymin and ymax
    # ax.set_zlim([-50, 50])  # Set appropriate values for zmin and zmax

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    plt.savefig('output/3d_points_house_loop.png')


if __name__ == '__main__':
    main()