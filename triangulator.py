import torch

class Triangulator:
    def __init__(self, mtx):
        """
        Constructor for Triangulator class.

        Parameters:
        - mtx: intrinsic camera matrix
        """
        self.mtx = mtx

    def triangulate(self, kp1, kp2, R, t, matches):
        # initialize lists to store the 3D points and their corresponding colors
        points_3d = []
        colors = []

        # loop over the matched keypoints
        for match in matches:
            # extract the indices of the matched keypoints
            idx1, idx2 = match.queryIdx, match.trainIdx

            # get the pixel coordinates of the matched keypoints
            x1, y1 = kp1[idx1].pt
            x2, y2 = kp2[idx2].pt

            # convert the pixel coordinates to homogeneous coordinates
            pt1 = torch.tensor([x1, y1, 1]).T
            pt2 = torch.tensor([x2, y2, 1]).T

            # compute the projection matrices of the two cameras
            P1 = self.mtx @ torch.eye(3, 4)
            P2 = self.mtx @ torch.cat((R, t), dim=1)

            # triangulate the 3D point using the two projection matrices
            pt3d_hom = torch.triangulate_points(P1, P2, pt1.unsqueeze(1), pt2.unsqueeze(1))

            # convert the homogeneous 3D point to Cartesian 3D point
            pt3d = pt3d_hom[:-1] / pt3d_hom[-1]

            # append the 3D point and its color to the corresponding lists
            points_3d.append(pt3d)
            colors.append(kp1[idx1].pt)

        # convert the lists to tensors
        points_3d = torch.stack(points_3d)
        colors = torch.tensor(colors)

        return points_3d, colors