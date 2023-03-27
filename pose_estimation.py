import numpy as np
import cv2

class PoseEstimator:
    def __init__(self, mtx, dist, method='ransac'):
        """
        Constructor for FeatureExtractor class.

        Parameters:
        - mtx: intrinsic camera matrix
        - dist: distortion coefficients
        - method: feature extraction method (default: 'sift')
        """
        self.mtx = mtx
        self.dist = dist
        self.method = method
    
    def estimate(self, kp1, kp2, matches):
        """
        Extracts features and descriptors from an image.

        Parameters:
        - img: input image

        Returns:
        - kp: list of keypoints
        - des: list of descriptors
        """
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        E, _ = cv2.findEssentialMat(pts1, pts2, self.mtx, method=self.method, threshold=0.999, prob=0.999)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.mtx, None, None, None)
        return R, t