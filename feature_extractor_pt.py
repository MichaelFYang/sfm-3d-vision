import torch
import kornia as K
import numpy as np

import matplotlib.pyplot as plt

from utils import visualize_LAF

class FeatureExtractor:
    def __init__(self, mtx, dist, method='sift'):
        if isinstance(mtx, np.ndarray):
            self.mtx = torch.tensor(mtx, dtype=torch.float32, requires_grad=True)
        else:
            self.mtx = mtx

        if isinstance(dist, np.ndarray):
            self.dist = torch.tensor(dist, dtype=torch.float32, requires_grad=True)
        else:
            self.dist = dist

        self.method = method
        
        if method == 'sift':
            # TODO: add to.(device) in the future
            # self.detector = K.feature.HarrisCornerDetector()
            self.patch_size = 41
            resp = K.feature.BlobDoG()
            self.descriptor = K.feature.SIFTDescriptor(self.patch_size)
            scale_pyr = K.geometry.ScalePyramid(min_size=self.patch_size, double_image=True)
            nms = K.geometry.ConvQuadInterp3d()
            n_features = 500
            # n_features = 5000
            self.detector = K.feature.ScaleSpaceDetector(n_features,
                                        resp_module=resp,
                                        scale_space_response=True,#We need that, because DoG operates on scale-space
                                        nms_module=nms,
                                        scale_pyr_module=scale_pyr,
                                        ori_module=K.feature.LAFOrienter(),
                                        aff_module=K.feature.LAFAffineShapeEstimator(),
                                        mr_size=6.0,
                                        minima_are_also_good=True) #dark blobs are as good as bright.



        if method == 'LoFTR':
            self.detector = K.feature.LoFTR(pretrained="outdoor")

    def undistort_image(self, img):
        img_tensor = K.image_to_tensor(img, keepdim=False).float()
        undistorted_image = K.geometry.calibration.undistort_image(img_tensor, self.mtx, self.dist)
        return undistorted_image

    def rgb_to_gray(self, img):
        gray_image = K.color.rgb_to_grayscale(img)
        return gray_image

    def compute_response_map(self, gray):
        response_map = self.detector(gray)
        return response_map

    def extract(self, img):
        img_undist = self.undistort_image(img)
        gray = self.rgb_to_gray(img_undist)
        # img_gray = K.tensor_to_image(gray)
        if self.method == 'sift':
            lafs, resps = self.detector(gray)
            patches = K.feature.extract_patches_from_pyramid(gray, lafs, self.patch_size)
            B, N, CH, H, W = patches.size()
            # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
            # So we need to reshape a bit :) 
            descs = self.descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)

            # visualize_LAF(gray, lafs)

            return lafs, descs[0]

        elif self.method == 'LoFTR':
            pass
        
        else:
            raise ValueError('Invalid feature extraction method: {}'.format(self.method))


class FeatureMatcher:
    def __init__(self, matcher='fginn'):
        """
        Constructor for FeatureMatcher class.

        Parameters:
        - matcher: feature matcher method (default: 'bf') - brute force
        """
        self.matcher = matcher

    def match(self, des1, des2, kp1, kp2):
        """
        Matches features between two sets of descriptors.

        Parameters:
        - des1: descriptors from first image
        - des2: descriptors from second image

        Returns:
        - matches: list of matches between keypoints
        """
        if self.matcher == 'fginn':
            return K.feature.match_fginn(des1, des2, kp1, kp2, mutual=True)
        else:
            raise ValueError('Invalid feature matcher method: {}'.format(self.matcher))
