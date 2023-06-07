import torch
from utils import compute_reprojection_error

class ImageAdjuster():

    def __init__(self, R_opt, T_opt, point3d_opt, K, src_pts, dst_pts, optimizer='adam'):
        """
        Constructor for BundleAdjuster class.

        Parameters:
        - R_opt: leaf node that requires grad for R
        - T_opt: leaf node that requires grad for T
        - point3d_opt: leaf node that requires grad for point3d
        - K: intrinsic matrix
        - src_pts: source points
        - dst_pts: destination points
        - optimizer: optimizer to use (adam or LBFGS)
        """
        self.R_opt = R_opt
        self.T_opt = T_opt
        self.point3d_opt = point3d_opt

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam([self.R_opt, self.T_opt, self.point3d_opt], lr=5e-3)
        elif optimizer == 'LBFGS':
            self.optimizer = torch.optim.LBFGS([self.R_opt, self.T_opt, self.point3d_opt], lr=0.1)
            
        self.loss = compute_reprojection_error
        self.K = K
        self.src_pts = src_pts
        self.dst_pts = dst_pts

    def adjust_step(self):
        """
        Perform one adjustment step.

        """
        if self.optimizer.__class__.__name__ == 'LBFGS':
            def closure():
                # clear gradients for this training step
                self.optimizer.zero_grad()

                # compute reprojection error
                reproj_2d_1, reproj_2d_2, err = self.loss(self.point3d_opt, self.src_pts, self.dst_pts, R=self.R_opt, T=self.T_opt, K=self.K)
                
                # update R, T, point3d
                err.backward()
                return err

            self.optimizer.step(closure)
            with torch.no_grad():
                reproj_2d_1, reproj_2d_2, err = self.loss(self.point3d_opt, self.src_pts, self.dst_pts, R=self.R_opt, T=self.T_opt, K=self.K)
        else:
            # clear gradients for this training step
            self.optimizer.zero_grad()

            # compute reprojection error
            reproj_2d_1, reproj_2d_2, err = self.loss(self.point3d_opt, self.src_pts, self.dst_pts, R=self.R_opt, T=self.T_opt, K=self.K) 
            
            # update R, T, point3d
            err.backward()
            self.optimizer.step()

        return reproj_2d_1, reproj_2d_2, err