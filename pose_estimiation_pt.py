import torch
import kornia as K

class PoseEstimator:
    def __init__(self, mtx, dist, method='ransac'):
        """
        Constructor for PoseEstimator class.

        Args:
            mtx (numpy.ndarray): Camera matrix.
            dist (numpy.ndarray): Distortion coefficients.
            method (str): Method used for estimating essential matrix (default: 'ransac').
        """
        self.mtx = mtx
        self.dist = dist
        self.method = method
        self.ransca = K.geometry.ransac.RANSAC(model_type='fundamental')

    def estimate(self, kp1, kp2, matches):
        """
        Estimate camera pose from matched keypoints.

        Args:
            kp1 (list): Keypoints from first image.
            kp2 (list): Keypoints from second image.
            matches (list): Matched keypoints between first and second images.

        Returns:
            numpy.ndarray: Rotation matrix.
            numpy.ndarray: Translation vector.
        """
        pts1 = torch.tensor([kp1[m.queryIdx].pt for m in matches], dtype=torch.float32).unsqueeze(1)
        pts2 = torch.tensor([kp2[m.trainIdx].pt for m in matches], dtype=torch.float32).unsqueeze(1)

        # Convert camera matrix to PyTorch tensor
        K = torch.tensor(self.mtx, dtype=torch.float32)

        # Estimate essential matrix and recover pose
        E = self.essential_matrix(pts1, pts2, K)
        R, t = self.recover_pose(E, pts1, pts2, K)

        return R.numpy(), t.numpy()

    def essential_matrix(self, pts1, pts2, K):
        """
        Compute the essential matrix using the normalized 8-point algorithm.
        This implementation assumes you have installed PyTorch.

        Args:
            pts1 (torch.Tensor): Matched points from the first image.
            pts2 (torch.Tensor): Matched points from the second image.
            K (torch.Tensor): Camera intrinsic matrix.

        Returns:
            torch.Tensor: Essential matrix.
        """
        # Normalize points
        K_inv = torch.inverse(K)
        pts1_normalized = torch.matmul(pts1, K_inv.t())
        pts2_normalized = torch.matmul(pts2, K_inv.t())

        # Compute the fundamental matrix
        F = self.compute_fundamental_matrix(pts1_normalized, pts2_normalized)

        # Compute the essential matrix
        essential_mat = torch.mm(K.t(), torch.mm(F, K))

        return essential_mat
    
    def compute_fundametal_matrix_kornia(self, pts1, pts2):
        Fm, inlier = self.ransca.forward(pts1, pts2)
        return Fm, inlier

    def compute_fundamental_matrix(self, pts1, pts2):
        num_points = pts1.shape[0]

        # Construct the A matrix
        A = torch.zeros((num_points, 9), dtype=torch.float32)
        for i in range(num_points):
            x1, y1 = pts1[i, 0, 0], pts1[i, 0, 1]
            x2, y2 = pts2[i, 0, 0], pts2[i, 0, 1]
            A[i] = torch.tensor([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

        # Compute the SVD of A
        U, S, Vt = torch.svd(A)

        # The last column of V is the nullspace, which is the solution to Af = 0
        F = Vt[-1].view(3, 3)

        # Enforce the rank-2 constraint on the fundamental matrix
        Uf, Sf, Vtf = torch.svd(F)
        Sf[-1] = 0.0
        F_rank2 = torch.matmul(Uf, torch.matmul(torch.diag(Sf), Vtf))

        return F_rank2
    
    def recover_pose(self, E, pts1, pts2, K):
        # Perform SVD on the essential matrix
        U, _, Vt = torch.svd(E)

        # Ensure that determinant(U) and determinant(Vt) are positive
        if torch.det(U) < 0:
            U = -U
        if torch.det(Vt) < 0:
            Vt = -Vt

        # Create the possible rotation and translation matrices
        W = torch.tensor([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]], dtype=torch.float32)

        R1 = torch.matmul(U, torch.matmul(W, Vt))
        R2 = torch.matmul(U, torch.matmul(W.t(), Vt))
        t = U[:, 2].unsqueeze(1)

        # Determine the correct rotation and translation by checking the cheirality condition
        valid_points = 0
        R, T = None, None
        points3D = None

        for possible_R, possible_t in [(R1, t), (R1, -t), (R2, t), (R2, -t)]:
            # Project points into the second camera coordinate system
            P1 = K @ torch.eye(3, 4, dtype=torch.float32)
            P2 = K @ torch.cat((possible_R, possible_t), dim=1)
            points3D = self.triangulate_points(P1, P2, pts1, pts2)

            # Check the cheirality condition
            valid_points_mask = points3D[:, 2] > 0
            num_valid_points = torch.sum(valid_points_mask)

            if num_valid_points > valid_points:
                valid_points = num_valid_points
                R, T = possible_R, possible_t

        return R, T, points3D

    def triangulate_points(self, P1, P2, pts1, pts2):
        num_points = pts1.shape[0]
        points3D = torch.zeros((num_points, 4), dtype=torch.float32)

        for i in range(num_points):
            A = torch.zeros((4, 4), dtype=torch.float32)
            A[0:2] = pts1[i, 0] * P1[2] - P1[0]
            A[2:4] = pts1[i, 1] * P1[2] - P1[1]

            B = torch.zeros((4, 4), dtype=torch.float32)
            B[0:2] = pts2[i, 0] * P2[2] - P2[0]
            B[2:4] = pts2[i, 1] * P2[2] - P2[1]

            C = torch.matmul(A, torch.inverse(B))

            # Perform SVD on C
            U, _, Vt = torch.svd(C)

            # The last column of V is the nullspace, which is the solution to Cx = 0
            X = Vt[-1]

            # Homogeneous to inhomogeneous coordinates
            X = X / X[-1]

            points3D[i] = X

        return points3D