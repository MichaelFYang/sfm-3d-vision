import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, mtx, dist, method='ransac'):
        """
        Constructor for PoseEstimator class.

        Args:
            mtx (numpy.ndarray): Camera matrix.
            dist (numpy.ndarray): Distortion coefficients.
            method (str): Method used for estimating essential matrix (default: 'ransac').
        """
        self.mtx = mtx
        self.dist = dist
        if method == 'ransac':
            self.method = cv2.RANSAC
    
    def estimate(self, kp1, kp2, matches):
        """
        Estimate camera pose from matched keypoints.

        Args:
            kp1 (list): Keypoints from first image.
            kp2 (list): Keypoints from second image.
            matches (list): Matched keypoints between first and second images.

        Returns:
            numpy.ndarray: Rotation matrix.
            numpy.ndarray: Translation vector.
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate essential matrix and recover pose
        E, mask = cv2.findEssentialMat(pts1, pts2, self.mtx, method=self.method, threshold=0.999, prob=0.999)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.mtx, mask)

        return R, t, mask
