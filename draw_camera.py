import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_camera_pose(T):
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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

    # Set plot limits and labels
    ax.set_xlim3d(-2.0, 2.0)
    ax.set_ylim3d(-2.0, 2.0)
    ax.set_zlim3d(-2.0, 2.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Pose')

    # Show the plot
    plt.show()

# Example usage
T = np.array([[5.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 3.0, 2.0],
              [0.0, 0.0, 0.0, 1.0]])

draw_camera_pose(T)
