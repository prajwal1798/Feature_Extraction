# =============================================================================
# DEEPONET: BRANCH / TRUNK OPERATOR NETWORK (Lu et al. 2021)
#
# The temperature at grid point x_j of case i is approximated by
#
#     T(x_j)  ~  sum_k  b_k(phi_i) * t_k(x_j, y_j, phi_i(x_j))  +  b_0
#
#   branch  b : R^{Ny*Nx} -> R^p   reads the whole SDF field phi_i (12288 values)
#   trunk   t : R^3       -> R^p   reads one point: (x/Lx, y/Ly, local SDF)
#   b_0       : one learned scalar bias
#   p         : latent dimension (number of basis functions), default 128
#
# Both networks are plain fully connected networks
#     z_{l+1} = sigma( W_l z_l + c_l ),   last layer linear (no sigma).
# Nothing else: no Fourier features, no residual paths, no convolutions.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   branch_in     : size of the branch input  (Ny*Nx = 12288)
#   branch_hidden : list of hidden-layer widths of the branch, e.g. [256, 256]
#   trunk_hidden  : list of hidden-layer widths of the trunk,  e.g. [128, 128]
#   latent        : p, the common output width of branch and trunk
#   activation    : "tanh" or "gelu"
#   trunk_in      : 3  (x/Lx, y/Ly, SDF)
#
#   tensor shapes:  branch input (B, 12288)   -> branch features (B, p)
#                   trunk input  (B, Q, 3)    -> trunk features  (B, Q, p)
#                   output       (B, Q)        B cases, Q points
# =============================================================================

import torch
from torch import nn


def activation_layer(name):
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("Activation must be tanh or gelu")


# =============================================================================
# FULLY CONNECTED NETWORK  sizes[0] -> sizes[1] -> ... -> sizes[-1]
#
# Activation after every layer except the last. Built as nn.Sequential, so
# the parameter names (0.weight, 2.weight, ...) match older checkpoints.
# =============================================================================
def dense_network(sizes, activation):
    layers = []
    n_layers = len(sizes) - 1

    for l in range(n_layers):
        layers.append(nn.Linear(sizes[l], sizes[l + 1]))
        if l < n_layers - 1:
            layers.append(activation_layer(activation))

    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, branch_in, branch_hidden, trunk_hidden, latent, activation="tanh",
                 trunk_in=3):
        super().__init__()

        # STORED WITH EVERY CHECKPOINT SO THE MODEL CAN BE REBUILT EXACTLY
        self.model_kind = "deeponet"
        self.architecture = {
            "branch_in": branch_in,
            "branch_hidden": list(branch_hidden),
            "trunk_hidden": list(trunk_hidden),
            "latent": latent,
            "activation": activation,
            "trunk_in": trunk_in,
        }

        branch_sizes = [branch_in] + list(branch_hidden) + [latent]
        trunk_sizes = [trunk_in] + list(trunk_hidden) + [latent]

        self.branch = dense_network(branch_sizes, activation)
        self.trunk = dense_network(trunk_sizes, activation)
        self.bias = nn.Parameter(torch.zeros(1))

    # =========================================================================
    # COMBINE:  T_ij = sum_k b_ik t_ijk + b_0
    #
    # The branch is evaluated once per case; the trunk may be evaluated on a
    # subset ("chunk") of the points to save memory (see training/runtime.py).
    # =========================================================================
    def forward_from_branch(self, branch_features, trunk_input):
        trunk_features = self.trunk(trunk_input)
        temperature = torch.einsum("bk,bnk->bn", branch_features, trunk_features)
        return temperature + self.bias

    def forward(self, branch_input, trunk_input):
        branch_features = self.branch(branch_input)
        return self.forward_from_branch(branch_features, trunk_input)
