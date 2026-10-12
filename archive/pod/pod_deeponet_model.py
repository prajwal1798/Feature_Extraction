"""Dense four-hidden-layer branch with a fixed training-derived POD trunk."""

import torch
from torch import nn
from .deeponet_model import dense_network


class PODDeepONet(nn.Module):
    def __init__(self, branch_in, branch_hidden, latent, activation="tanh"):
        super().__init__()
        if latent < 1 or latent > branch_in:
            raise ValueError("Invalid POD rank")
        self.model_kind = "pod_deeponet"
        self.architecture = {
            "branch_in": branch_in, "branch_hidden": list(branch_hidden),
            "latent": latent, "activation": activation,
        }
        self.branch = dense_network(
            [branch_in] + list(branch_hidden) + [latent], activation
        )
        # Start at the training mean field; keep hidden-layer initialization.
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)
        self.register_buffer("pod_mean", torch.zeros(branch_in))
        self.register_buffer("pod_basis", torch.zeros(branch_in, latent))

    def set_basis(self, mean, basis):
        if tuple(mean.shape) != tuple(self.pod_mean.shape) or tuple(basis.shape) != tuple(self.pod_basis.shape):
            raise ValueError("POD basis dimensions do not match model")
        with torch.no_grad():
            self.pod_mean.copy_(torch.as_tensor(mean))
            self.pod_basis.copy_(torch.as_tensor(basis))

    def forward_from_branch(self, coefficients, first=0, last=None):
        return coefficients @ self.pod_basis[first:last].T + self.pod_mean[first:last]

    def forward(self, sdf):
        return self.forward_from_branch(self.branch(sdf))
