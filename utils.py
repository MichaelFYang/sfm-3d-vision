import numpy as np
import pypose as pp
import kornia as K
import matplotlib.pyplot as plt
import cv2
import math
import torch
from mpl_toolkits.mplot3d import Axes3D

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

def get_noise_rotation(noise_std_dev):
    noise_rad = torch.normal(mean=0., std=noise_std_dev, size=(3,))  # in radians
    noise_rad_x, noise_rad_y, noise_rad_z = noise_rad[0], noise_rad[1], noise_rad[2]
    
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(noise_rad_x), -math.sin(noise_rad_x)],
        [0, math.sin(noise_rad_x), math.cos(noise_rad_x)],
    ])

    Ry = torch.tensor([
        [math.cos(noise_rad_y), 0, math.sin(noise_rad_y)],
        [0, 1, 0],
        [-math.sin(noise_rad_y), 0, math.cos(noise_rad_y)],
    ])

    Rz = torch.tensor([
        [math.cos(noise_rad_z), -math.sin(noise_rad_z), 0],
        [math.sin(noise_rad_z), math.cos(noise_rad_z), 0],
        [0, 0, 1],
    ])

    noise_rotation_matrix = Rz.matmul(Ry).matmul(Rx)

    return noise_rotation_matrix


#Lets define some functions for local feature matching
def visualize_LAF(img, LAF, img_idx = 0):
    x, y = K.feature.laf.get_laf_pts_to_draw(LAF, img_idx)
    plt.figure()
    plt.imshow(K.utils.tensor_to_image(img[img_idx]))
    plt.plot(x, y, 'r')
    plt.show()
    return

def compute_reprojection_error(point3d, src_pts, dst_pts, R=None, T=None, K=None, P=None):
    """
    Input:
        point3d: array of triangulated 3D points in homo coordinate (N, 4)
        P: projection matrix of camera 2 (3, 4)
        src_pts: original 2D image points of camera 1 (N, 2)
        dst_pts: original 2D image points of camera 2 (N, 2)
    """
    
    """
    point3D = torch.cat([point3D, torch.tensor([1.]).float()], dim=0)  # Convert to homogeneous coordinates
    point2D_proj = K @ (R @ point3D + T)
    point2D_proj = point2D_proj / point2D_proj[2]  # Convert back to inhomogeneous coordinates
    return torch.norm(point2D - point2D_proj[:2])
    """
    # import pdb; pdb.set_trace()
    N = point3d.shape[0]
    point3d = torch.hstack((point3d, torch.ones((N, 1))))  # Convert to homogeneous coordinates
    
    P1 = K @ torch.hstack((torch.eye(3), torch.zeros((3,1))))
    if R is not None and T is not None and K is not None:
        P2 = K @ torch.hstack((R, T.reshape((3,1))))
        reproj_2d_2 = point3d @ P2.T
    else:
        if pp.is_lietensor(P):
            reproj_2d_2 = P @ point3d
            reproj_2d_2 = (K @ reproj_2d_2[:,:-1].T).T
        else:
            reproj_2d_2 = (K @ P @ point3d.T).T

    reproj_2d_1 = point3d @ P1.T 
    

    reproj_2d_1 = reproj_2d_1 / reproj_2d_1[:, -1].unsqueeze(1)
    reproj_2d_2 = reproj_2d_2 / reproj_2d_2[:, -1].unsqueeze(1)

    reproj_2d_1 = reproj_2d_1[:, :2]
    reproj_2d_2 = reproj_2d_2[:, :2]

    distance_1 = torch.sum(torch.norm(reproj_2d_1 - src_pts, p=2, dim=1))
    distance_2 = torch.sum(torch.norm(reproj_2d_2 - dst_pts, p=2, dim=1))

    return reproj_2d_1, reproj_2d_2, (distance_1 + distance_2)/(2*N)

def visualize_reprojection(img1, img2, src_pts, dst_pts, reproj_2d_1, reproj_2d_2):
    """Visualize the reprojection errors for multiple points on the image."""
    # Create a figure and set up subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img1)
    axs[1].imshow(img2)

    # Convert tensors to numpy arrays for visualization
    src_pts = src_pts.detach().numpy()
    dst_pts = dst_pts.detach().numpy()
    reproj_2d_1 = reproj_2d_1.detach().numpy()
    reproj_2d_2 = reproj_2d_2.detach().numpy()
    
    for i, (points2d, points2d_proj) in enumerate([(src_pts, reproj_2d_1),(dst_pts, reproj_2d_2)]):
        # Draw the original and reprojected points
        axs[i].plot(points2d[:, 0], points2d[:, 1], 'bo')  # Original point in blue
        axs[i].plot(points2d_proj[:, 0], points2d_proj[:, 1], 'ro')  # Reprojected point in red

        # Draw a line between original and reprojected points
        axs[i].plot([points2d[:, 0], points2d_proj[:, 0]], [points2d[:, 1], points2d_proj[:, 1]], 'g-')  # Error line in green
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig('output/reproj_err.png')
    plt.close()
    # plt.show()

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

def draw_camera_pose(ax, T):
    """
    Draw camera pose using the transformation matrix.

    Args:
        T (numpy.ndarray): 4x4 transformation matrix representing the camera pose.

    Returns:
        None
    """
    # Camera intrinsic parameters
    fx = 1.0  # Focal length along x-axis
    fy = 1.0  # Focal length along y-axis
    cx = 0.0  # Principal point x-coordinate
    cy = 0.0  # Principal point y-coordinate

    # Camera frustum vertices
    vertices = np.array([[0.5, 0.5, 1.0],
                         [-0.5, 0.5, 1.0],
                         [-0.5, -0.5, 1.0],
                         [0.5, -0.5, 1.0],
                         [0.0, 0.0, 0.0]])

    # Camera frustum in homogeneous coordinates
    frustum_homogeneous = np.hstack((vertices, np.ones((5, 1))))

    # Transform camera frustum to world coordinates
    frustum_world = np.dot(T, frustum_homogeneous.T).T

    # Extract camera center
    camera_center = frustum_world[-1, :3]

    # Plot camera frustum
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    ax.plot([frustum_world[0, 0], frustum_world[1, 0]],
            [frustum_world[0, 1], frustum_world[1, 1]],
            [frustum_world[0, 2], frustum_world[1, 2]],
            'r-', linewidth=2)
    ax.plot([frustum_world[1, 0], frustum_world[2, 0]],
            [frustum_world[1, 1], frustum_world[2, 1]],
            [frustum_world[1, 2], frustum_world[2, 2]],
            'r-', linewidth=2)
    ax.plot([frustum_world[2, 0], frustum_world[3, 0]],
            [frustum_world[2, 1], frustum_world[3, 1]],
            [frustum_world[2, 2], frustum_world[3, 2]],
            'r-', linewidth=2)
    ax.plot([frustum_world[3, 0], frustum_world[0, 0]],
            [frustum_world[3, 1], frustum_world[0, 1]],
            [frustum_world[3, 2], frustum_world[0, 2]],
            'r-', linewidth=2)
    ax.plot([frustum_world[0, 0], frustum_world[4, 0]],
            [frustum_world[0, 1], frustum_world[4, 1]],
            [frustum_world[0, 2], frustum_world[4, 2]],
            'r-', linewidth=2)
    ax.plot([frustum_world[1, 0], frustum_world[4, 0]],
            [frustum_world[1, 1], frustum_world[4, 1]],
            [frustum_world[1, 2], frustum_world[4, 2]],
            'r-', linewidth=2)
    ax.plot([frustum_world[2, 0], frustum_world[4, 0]],
            [frustum_world[2, 1], frustum_world[4, 1]],
            [frustum_world[2, 2], frustum_world[4, 2]],
            'r-', linewidth=2)
    ax.plot([frustum_world[3, 0], frustum_world[4, 0]],
            [frustum_world[3, 1], frustum_world[4, 1]],
            [frustum_world[3, 2], frustum_world[4, 2]],
            'r-', linewidth=2)

    # Plot camera coordinate axes
    axes_length = 0.2
    x_axis = np.dot(T, np.array([[0.0, 0.0, 0.0, 1.0],
                                 [axes_length, 0.0, 0.0, 1.0]]).T).T
    y_axis = np.dot(T, np.array([[0.0, 0.0, 0.0, 1.0],
                                 [0.0, axes_length, 0.0, 1.0]]).T).T
    z_axis = np.dot(T, np.array([[0.0, 0.0, 0.0, 1.0],
                                 [0.0, 0.0, axes_length, 1.0]]).T).T

    ax.plot([camera_center[0], x_axis[1, 0]],
            [camera_center[1], x_axis[1, 1]],
            [camera_center[2], x_axis[1, 2]],
            'r-', linewidth=2)
    ax.plot([camera_center[0], y_axis[1, 0]],
            [camera_center[1], y_axis[1, 1]],
            [camera_center[2], y_axis[1, 2]],
            'g-', linewidth=2)
    ax.plot([camera_center[0], z_axis[1, 0]],
            [camera_center[1], z_axis[1, 1]],
            [camera_center[2], z_axis[1, 2]],
            'b-', linewidth=2)

    # # Set plot limits and labels
    # ax.set_xlim3d(-2.0, 2.0)
    # ax.set_ylim3d(-2.0, 2.0)
    # ax.set_zlim3d(-2.0, 2.0)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Camera Pose')

    # Show the plot
    # plt.show()