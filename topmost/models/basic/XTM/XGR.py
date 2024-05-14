import torch
from torch import nn


class XGR(nn.Module):
    def __init__(self, weight_loss_XGR, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_XGR = weight_loss_XGR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def symmetric_sinkhorn(C, eps=1e0, f0=None, max_iter=1000, tol=1e-6, verbose=False):
        """
        Performs Sinkhorn iterations in log domain to solve the entropic symmetric
        OT problem with symmetric cost C and entropic regularization eps.
        """
        n = C.shape[0]
        
        if f0 is None:
            f = torch.zeros(n, dtype=C.dtype, device=C.device)
        else:
            f = f0

        for k in range(max_iter):
            # well-conditioned symmetric Sinkhorn update
            f = 0.5 * (f - eps * torch.logsumexp((f - C) / eps, -1))

            # check convergence every 10 iterations
            if k % 10 == 0: 
                log_T = (f[:, None] + f[None, :] - C) / eps
                if (torch.abs(torch.exp(torch.logsumexp(log_T, -1))-1) < tol).all():
                    if verbose:
                        print(f'---------- Breaking at iter {k} ----------')
                    break

            if k == max_iter-1:
                if verbose:
                    print('---------- Max iter attained for Sinkhorn algorithm ----------')

        return (f[:, None] + f[None, :] - C) / eps, f


    def forward(self, M, group, eps=1):
        log_Q, f = self.symmetric_sinkhorn(M, eps=eps, max_iter=1000, f0=f.detach())
        loss_XGR = (group * (group.log() - log_Q - 1) + torch.exp(log_Q)).sum()
        loss_XGR *= self.weight_loss_XGR

        return loss_XGR
