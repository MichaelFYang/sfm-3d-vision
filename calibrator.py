import numpy as np
import cv2

class CameraCalibrator:
    def __init__(self, pattern_size=(9, 6)):
        """
        Constructor for Calibrator class.

        Parameters:
        - pattern_size: tuple of number of inner corners in the checkerboard pattern
        """
        self.pattern_size = pattern_size
    
    def calibrate(self, imgs):
        """
        Calibrates a camera using a set of calibration images.

        Parameters:
        - img_list: list of calibration images

        Returns:
        - mtx: intrinsic camera matrix
        - dist: distortion coefficients
        """
        objp = np.zeros((self.pattern_size[0]*self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []
        gray_imgs = []
        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                gray_imgs.append(gray)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_imgs[0].shape[::-1], None, None)
        return mtx, dist