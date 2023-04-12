import numpy as np
import cv2

class Triangulator:
    def __init__(self, mtx):
        """
        Constructor for Triangulator class.

        Parameters:
        - mtx: intrinsic camera matrix
        """
        self.mtx = mtx
    
    def triangulate(self, kp1, kp2, R, t, matches, matchesMask):
        """
        Triangulates 3D points from a set of correspondences and camera poses.

        Parameters:
        - kp1: keypoints in first image
        - kp2: keypoints in second image
        - R: rotation matrix from second to first camera
        - t: translation vector from second to first camera
        - matches: list of matches between keypoints

        Returns:
        - pts_3d: list of 3D points
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        pts1 = pts1[np.asarray(matchesMask) == 1, :, :]
        pts2 = pts2[np.asarray(matchesMask) == 1, :, :]

        proj_mat1 = np.dot(self.mtx, np.hstack((np.eye(3), np.zeros((3, 1)))))
        proj_mat2 = np.dot(self.mtx, np.hstack((R, t)))

        proj_pts1 = cv2.undistortPoints(pts1, self.mtx, None)
        proj_pts2 = cv2.undistortPoints(pts2, self.mtx, None)

        proj_pts1 = np.squeeze(proj_pts1).T
        proj_pts2 = np.squeeze(proj_pts2).T

        pts_4d_hom = cv2.triangulatePoints(proj_mat1, proj_mat2, proj_pts1, proj_pts2)
        pts_3d_hom = pts_4d_hom / pts_4d_hom[3]
        pts_3d = pts_3d_hom[:3].T
        return pts_3d