import torch
import kornia as K

class FeatureExtractor:
    def __init__(self, mtx, dist, method='sift'):
        self.mtx = torch.tensor(mtx, dtype=torch.float32)
        self.dist = torch.tensor(dist, dtype=torch.float32)
        self.method = method
        self.sift_descriptor = K.features.SIFTDescriptor(32, 8)

        if method == 'sift':
            self.detector = K.features.HarrisCornerDetector()

    def undistort_image(self, img):
        img_tensor = K.image_to_tensor(img, keepdim=False).float()
        undistorted_image = K.remove_lens_distortion(img_tensor, self.mtx, self.dist)
        return undistorted_image

    def rgb_to_gray(self, img):
        gray_image = K.rgb_to_grayscale(img)
        return gray_image

    def extract(self, img):
        img_undist = self.undistort_image(img)
        gray = self.rgb_to_gray(img_undist)

        if self.method == 'sift':
            # Detect keypoints
            keypoints = self.detector(gray)

            # Compute orientations
            orientations = K.features.orientation(gray, keypoints)

            # Compute SIFT descriptors
            descriptors = self.sift_descriptor(gray, keypoints, orientations)
            return keypoints, descriptors
        else:
            raise ValueError('Invalid feature extraction method: {}'.format(self.method))

class FeatureMatcher:
    def __init__(self, matcher='bf'):
        """
        Constructor for FeatureMatcher class.

        Parameters:
        - matcher: feature matcher method (default: 'bf') - brute force
        """
        self.matcher = matcher

    def match(self, des1, des2):
        """
        Matches features between two sets of descriptors.

        Parameters:
        - des1: descriptors from first image
        - des2: descriptors from second image

        Returns:
        - matches: list of matches between keypoints
        """
        if self.matcher == 'bf':
            return self.match_brute_force(des1, des2)
        else:
            raise ValueError('Invalid feature matcher method: {}'.format(self.matcher))

    def match_brute_force(self, des1, des2):
        """
        Brute force matching of features between two sets of descriptors using L2 distance.

        Parameters:
        - des1: descriptors from first image
        - des2: descriptors from second image

        Returns:
        - matches: list of matches between keypoints
        """
        distance_matrix = K.losses.pairwise_distance(des1, des2)
        matches = torch.argmin(distance_matrix, dim=1)

        return matches