import cv2
import torch
from typing import Callable, Tuple
from functools import partial


FeatureMatchingFunc = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,torch.Tensor]]

def NearestNeighbor_torch(des1: torch.Tensor, des2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if (len(des1)==0 or len(des2)==0):
        print('No match...')
        return
    dist_mat = torch.cdist(des1, des2, p=2)
    match_dists, idxs_in_2 = torch.min(dist_mat, dim=1)
    idxs_in_1 = torch.arange(0, idxs_in_2.size(0), device=idxs_in_2.device)
    matches_idxs = torch.cat([idxs_in_1.view(-1,1), idxs_in_2.view(-1,1)], 1)
    return match_dists.view(-1,1), matches_idxs.view(-1,1)

class FeatureExtractor:
    def __init__(self, mtx, dist, method='sift'):
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
    
    def extract(self, img):
        """
        Extracts features and descriptors from an image.

        Parameters:
        - img: input image

        Returns:
        - kp: list of keypoints
        - des: list of descriptors
        """
        # Undistort input image
        img_undist = cv2.undistort(img, self.mtx, self.dist)

        # Convert image to grayscale
        gray = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        if self.method == 'sift':
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(gray, None)
        elif self.method == 'surf':
            surf = cv2.xfeatures2d.SURF_create()
            kp, des = surf.detectAndCompute(gray, None)
        else:
            raise ValueError('Invalid feature extraction method: {}'.format(self.method))

        return kp, des

class FeatureMatcher:
    def __init__(self, matcher='bf'):
        """
        Constructor for FeatureMatcher class.

        Parameters:
        - matcher: feature matcher method (default: 'bf') - brute force
        """
        if matcher == 'bf':
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        elif matcher == 'flann':
            index_params = dict(algorithm=0, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def match(self, des1, des2):
        """
        Matches features between two sets of descriptors.

        Parameters:
        - des1: descriptors from first image
        - des2: descriptors from second image

        Returns:
        - matches: list of matches between keypoints
        """
        matches = self.matcher.match(des1, des2)
        # matches = self.matcher.knnMatch(des1, des2, k=2)

        return matches
    
class FeatureMatcher_torch:

    match: FeatureMatchingFunc

    def __init__(self, matcher="nn"):
        r"""
        FeatureMatcher_torch class constructor.

        Parameters:
        - matcher: feature matcher method (default: 'nn' - nearest neighbor)
        """
        if matcher == 'nn':
            self.match = NearestNeighbor_torch