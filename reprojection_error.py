import torch

def compute_reprojection_error(point3d, R, T, K, src_pts, dst_pts):
    """
    Input:
        point3d: array of triangulated 3D points in homo coordinate (N, 4)
        P1: projection matrix of camera 1 (3, 4)
        P2: projection matrix of camera 2 (3, 4)
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
    P2 = K @ torch.hstack((R, T.reshape((3,1))))

    reproj_2d_1 = point3d @ P1.T 
    reproj_2d_2 = point3d @ P2.T

    reproj_2d_1 = reproj_2d_1 / reproj_2d_1[:, -1].unsqueeze(1)
    reproj_2d_2 = reproj_2d_2 / reproj_2d_2[:, -1].unsqueeze(1)

    reproj_2d_1 = reproj_2d_1[:, :2]
    reproj_2d_2 = reproj_2d_2[:, :2]

    distance_1 = torch.sum(torch.norm(reproj_2d_1 - src_pts, dim=1))/N
    distance_2 = torch.sum(torch.norm(reproj_2d_2 - dst_pts, dim=1))/N

    return reproj_2d_1, reproj_2d_2, distance_1 + distance_2