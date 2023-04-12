import cv2
import numpy as np
import matplotlib.pyplot as plt
from calibrator import CameraCalibrator
from feature_extractor import FeatureExtractor, FeatureMatcher
from triangulator import Triangulator
from pose_estimation import PoseEstimator

def main():
    # Define calibration pattern size
    # pattern_size = (9, 6)
    # pattern_size = (23, 31)

    # # Load calibration images
    # img_list = [cv2.imread('calibration_images/camera_calib{}.jpg'.format(i+1)) for i in range(10)]
    #
    # # Calibrate camera
    # calibrator = CameraCalibrator(pattern_size)
    #
    # # return intrinsic matrix and distortion coefficients
    # mtx, dist = calibrator.calibrate(img_list)

    # mannually input K
    mtx = np.array([[518.86, 0., 285.58],
                  [0., 519.47, 213.74],
                  [0., 0., 1.]])
    dist = np.zeros((5,))

    # Initialize feature extractor and feature matcher
    feature_extractor = FeatureExtractor(mtx, dist, method='sift')
    feature_matcher = FeatureMatcher(matcher='flann')

    # Load images
    img1 = cv2.imread('images/a1.png')
    img2 = cv2.imread('images/a2.png')

    # Extract features and descriptors
    '''
    Keypoints correspond to specific locations in the image. cv2.KeyPoint
    Each keypoint is represented by a 2D coordinate (x, y) and a scale and orientation.

    Descriptors are vectors that describe the local appearance of the region around each keypoint.
    numpy array of size (num_keypoints x descriptor_size)
    '''
    kp1, des1 = feature_extractor.extract(img1)
    kp2, des2 = feature_extractor.extract(img2)

    # Match features
    matches = feature_matcher.match(des1, des2)

    # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)
    # matches = good

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)

    img_siftmatch = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imshow('SIFT_Matches', img_siftmatch)

    # essential matrix
    pose_estimator = PoseEstimator(mtx, dist)
    R, t, mask = pose_estimator.estimate(kp1, kp2, matches)

    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img_inliermatch = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    cv2.imshow('Inlier_Matches', img_inliermatch)

    # Triangulate 3D points
    triangulator = Triangulator(mtx)
    pts_3d = triangulator.triangulate(kp1, kp2, R, t, matches, matchesMask)

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
