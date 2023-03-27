import cv2

class FeatureExtractor:
    def __init__(self, detector='sift'):
        """
        Constructor for FeatureExtractor class.

        Parameters:
        - detector: feature detector method (default: 'sift')
        """
        if detector == 'sift':
            self.detector = cv2.xfeatures2d.SIFT_create()
        elif detector == 'surf':
            self.detector = cv2.xfeatures2d.SURF_create()
        elif detector == 'orb':
            self.detector = cv2.ORB_create()
    
    def extract(self, img):
        """
        Extracts features and descriptors from an image.

        Parameters:
        - img: input image

        Returns:
        - kp: list of keypoints
        - des: list of descriptors
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
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
        return matches