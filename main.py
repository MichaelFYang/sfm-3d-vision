import cv2
import numpy as np
import matplotlib.pyplot as plt
from calibrator import CameraCalibrator
from feature_extractor import FeatureExtractor, FeatureMatcher
from triangulator import Triangulator

def main():
    # Define calibration pattern size
    pattern_size = (9, 6)

    # Load calibration images
    img_list = [cv2.imread('calibration_images/calibration{}.jpg'.format(i+1)) for i in range(10)]

    # Calibrate camera
    calibrator = CameraCalibrator(pattern_size)
    mtx, dist = calibrator.calibrate(img_list)

    # Initialize feature extractor and feature matcher
    feature_extractor = FeatureExtractor()
    feature_matcher = FeatureMatcher()

    # Load images
    img1 = cv2.imread('images/img1.jpg')
    img2 = cv2.imread('images/img2.jpg')

    # Extract features and descriptors
    kp1, des1 = feature_extractor.extract(img1)
    kp2, des2 = feature_extractor.extract(img2)

    # Match features
    matches = feature_matcher.match(des1, des2)

    # Triangulate 3D points
    R = np.eye(3)
    t = np.zeros((3, 1))
    triangulator = Triangulator(mtx)
    pts_3d = triangulator.triangulate(kp1, kp2, R, t, matches)

    # Visualize 3D points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == '__main__':
    main()
    
    
'''
1. Load calibration images and define calibration pattern size
2. Calibrate camera using the Calibrator class
3. Initialize feature extractor and feature matcher using the FeatureExtractor and FeatureMatcher classes
4. Load images to be used for SfM
5. Extract features and descriptors from the images using the extract method of the FeatureExtractor class
6. Match features between the images using the match method of the FeatureMatcher class
7. Triangulate 3D points from the matched features using the Triangulator class and the camera calibration parameters
8. Print the number of 3D points that were triangulated
'''
