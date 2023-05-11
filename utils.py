import numpy as np
import kornia as K
import matplotlib.pyplot as plt
import cv2

def get_pinhole_intrinsic_params(calibration_file_dir):
    """
    Calculate intrinsic matrix of the camera from file

    Args:
        the directory path which contains intrinsic paramters

    Returns:
        intrinsic matrix of the camera
    """

    K = []
    with open(calibration_file_dir + '/calibration.txt') as f:
        lines = f.readlines()
        calib_info = [float(val) for val in lines[0].split(' ')]
        row1 = [calib_info[0], 0, calib_info[2]]
        row2 = [0, calib_info[1], calib_info[3]]
        row3 = [0, 0, 1]

        K.append(row1)
        K.append(row2)
        K.append(row3)

        K = np.array(K, dtype=np.float)
    return K

#Lets define some functions for local feature matching
def visualize_LAF(img, LAF, img_idx = 0):
    x, y = K.feature.laf.get_laf_pts_to_draw(LAF, img_idx)
    plt.figure()
    plt.imshow(K.utils.tensor_to_image(img[img_idx]))
    plt.plot(x, y, 'r')
    plt.show()
    return


# drawMatches numpy version
def draw_matches(img1, kp1, img2, kp2, matches, inliers): 
    """Draws lines between matching keypoints of two images.  
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles 
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: ndarray [n1, 2]
        img2: An openCV image ndarray of the same format and with the same 
        element type as img1.
        kp2: ndarray [n2, 2]
        matches: ndarray [n_match, 2]
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.  
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.  
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 1
    thickness = 2
    color = (0, 255, 0)
    for i, m in enumerate(matches):
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        if inliers[i] == True:
            color = (0, 255, 0)
        else:
            continue
            color = (255, 0, 0)
        end1 = tuple(np.round(kp1[m[0]]).astype(int))
        end2 = tuple(np.round(kp2[m[1]]).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, color, thickness)
        cv2.circle(new_img, end1, r, color, thickness)
        cv2.circle(new_img, end2, r, color, thickness)
    return new_img
