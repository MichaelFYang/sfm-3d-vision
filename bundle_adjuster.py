import numpy as np
import cv2
from scipy import optimize
    
class BundleAdjuster:
    def init(self, mtx, dist):
        """
        Constructor for BundleAdjuster class.

        Parameters:
        - mtx: intrinsic camera matrix
        - dist: distortion coefficients
        """
        self.mtx = mtx
        self.dist = dist
    
    def adjust(self, poses, pts, kp_list, matches_list):
        """
        Performs bundle adjustment on a set of camera poses and 3D points.

        Parameters:
        - poses: list of camera poses as rotation matrices and translation vectors
        - pts: list of 3D points
        - kp_list: list of keypoints for each image
        - matches_list: list of matches for each image

        Returns:
        - proj_mat_list: list of optimized projection matrices
        - reprojection_error: reprojection error after bundle adjustment
        """
        objpoints = []
        imgpoints = []
        for i, (kp1, kp2, matches) in enumerate(matches_list):
            pts_3d = pts[i]
            pts_2d = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            objpoints.append(pts_3d)
            imgpoints.append(pts_2d)
        objpoints = np.array(objpoints)
        imgpoints = np.array(imgpoints)
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(objpoints, imgpoints, self.mtx, self.dist)
        proj_mat_list = [np.hstack((cv2.Rodrigues(r)[0], t)) for r, t in zip(rvecs, tvecs)]
        proj_pts_list = [cv2.undistortPoints(kp, self.mtx, self.dist) for kp in kp_list]
        reprojection_error = self.bundle_adjust(proj_mat_list, proj_pts_list, objpoints, imgpoints)
        return proj_mat_list, reprojection_error

    def bundle_adjust(self, proj_mat_list, proj_pts_list, objpoints, imgpoints):
        """
        Performs bundle adjustment on a set of camera poses and 3D points.

        Parameters:
        - proj_mat_list: list of initial projection matrices
        - proj_pts_list: list of undistorted keypoints for each image
        - objpoints: list of 3D points
        - imgpoints: list of image points

        Returns:
        - reprojection_error: reprojection error after bundle adjustment
        """
        
        def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
            """Function to minimize (i.e. reprojection error)."""
            """
            Function to minimize (i.e. reprojection error).

            Parameters:
            - params: flattened array of projection matrices and 3D points
            - n_cameras: number of cameras
            - n_points: number of 3D points
            - camera_indices: list of camera indices for each 3D point
            - point_indices: list of point indices for each camera
            - points_2d: list of image points

            Returns:
            - error: reprojection error
            """
            proj_mats = np.reshape(params[:n_cameras*12], (n_cameras, 3, 4))
            points_3d = np.reshape(params[n_cameras*12:], (n_points, 3))
            points_3d_hom = np.hstack((points_3d, np.ones((n_points, 1))))
            proj_points_2d_hom = np.zeros((points_3d_hom.shape[0], 3))
            for i in range(n_cameras):
                proj_points_3d_hom = np.dot(points_3d_hom[point_indices==i], proj_mats[i].T)
                proj_points_2d_hom[point_indices==i] = proj_points_3d_hom[:, :2] / proj_points_3d_hom[:, 2:]
            proj_points_2d = proj_points_2d_hom[:, :2]
            error = np.ravel(proj_points_2d - points_2d)
            return error
        
        n_cameras = len(proj_mat_list)
        n_points = objpoints.shape[0]
        camera_indices = np.arange(n_cameras).repeat(objpoints.shape[0])
        point_indices = np.tile(np.arange(objpoints.shape[0]), n_cameras)
        points_2d = np.array([kp.reshape(-1) for kp in proj_pts_list])
        # Flatten projection matrices and 3D points for optimization
        params = np.hstack((np.ravel(proj_mat_list), np.ravel(objpoints)))
        # Set up optimization problem
        cost_func = lambda x: fun(x, n_cameras, n_points, camera_indices, point_indices, points_2d)
        options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        res = optimize.least_squares(cost_func, params, jac='3-point', method='trf', options=options)
        # Extract optimized projection matrices and 3D points
        proj_mat_list = np.reshape(res.x[:n_cameras*12], (n_cameras, 3, 4))
        pts_3d = np.reshape(res.x[n_cameras*12:], (n_points, 3))
        # Compute reprojection error
        pts_3d_hom = np.hstack((pts_3d, np.ones((n_points, 1))))
        proj_pts_list = [cv2.projectPoints(pts_3d_hom[point_indices==i], cv2.Rodrigues(proj_mat_list[i][:, :3])[0], proj_mat_list[i][:, 3:], self.mtx, self.dist)[0] for i in range(n_cameras)]
        reprojection_error = np.linalg.norm(np.vstack([proj_pts.reshape(-1, 2) - imgpts.reshape(-1, 2) for proj_pts, imgpts in zip(proj_pts_list, imgpoints)]))
        return reprojection_error
