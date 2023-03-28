import cv2

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
        return matches