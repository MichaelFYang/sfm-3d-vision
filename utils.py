import numpy as np

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