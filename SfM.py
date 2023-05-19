from feature_extractor_pt import FeatureExtractor, FeatureMatcher
from pose_estimiation_pt import PoseEstimator
import kornia as K
import torch
import numpy as np
from utils import compute_reprojection_error


class StructurefromMotion:
    def __init__(self, mtx, dist):
        """
        Constructor for SfM class.

        Parameters:
        - mtx: intrinsic matrix
        - dist: distortion paramters
        """
        if isinstance(mtx, np.ndarray):
            self.mtx = torch.tensor(mtx, dtype=torch.float32, requires_grad=True)
        else:
            self.mtx = mtx
        
        if isinstance(dist, np.ndarray):
            self.dist = torch.tensor(dist, dtype=torch.float32, requires_grad=True)
        else:
            self.dist = dist

        # Initialize feature extractor and feature matcher
        self.feature_extractor = FeatureExtractor(mtx, dist, method='sift')
        self.feature_matcher = FeatureMatcher()
        self.pose_estimator = PoseEstimator(mtx, dist)

    def forward(self, img1, img2):
        """
        Perform structure from motion on a pair of images

        Parameters:
        - img1: 
        - img2: 

        Returns:
        - R
        - T
        - point3d
        - err
        - src_pts
        - dst_pts
        - reproj_2d_1
        - reproj_2d_2
        """
        kp1, des1 = self.feature_extractor.extract(img1)
        kp2, des2 = self.feature_extractor.extract(img2)
        
        # scores, matches = K.feature.match_snn(des1, des2, 0.9)
        scores, matches = self.feature_matcher.match(des1, des2, kp1, kp2)
        # scores, matches = K.feature.match_fginn(des1, des2, kp1, kp2, mutual=True)

        # Now RANSAC
        src_pts = kp1[0, matches[:,0], :, 2].float()
        dst_pts = kp2[0, matches[:,1], :, 2].float()

        Fm, inliers = self.pose_estimator.compute_fundametal_matrix_kornia(src_pts, dst_pts)
        src_pts = src_pts[inliers]
        dst_pts = dst_pts[inliers]

        Em = K.geometry.essential_from_fundamental(Fm, self.mtx, self.mtx)

        R, T, point3d = self.pose_estimator.recover_pose(Em, src_pts, dst_pts, self.mtx)
        # R, T, point3d = K.geometry.epipolar.motion_from_essential_choose_solution(Em, mtx_torch, mtx_torch, src_pts, dst_pts, mask=None)

        reproj_2d_1, reproj_2d_2, err = compute_reprojection_error(point3d, src_pts, dst_pts, R=R, T=T, K=self.mtx)
        
        return err, R, T, point3d, src_pts, dst_pts, reproj_2d_1, reproj_2d_2