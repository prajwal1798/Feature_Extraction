# =============================================================================
# FNO: TWO-DIMENSIONAL FOURIER NEURAL OPERATOR (Li et al. 2021)
#
# Input : 3 channels on the grid  (SDF, x/Lx, y/Ly),       shape (B, 3, Ny, Nx)
# Output: 1 channel, scaled temperature,                   shape (B, 1, Ny, Nx)
#
#   v_0     = P a                              lift: 1x1 convolution 3 -> width
#   (zero-pad the grid by "padding" points on the top and right)
#   v_{l+1} = sigma( K_l v_l + W_l v_l ),      l = 0 .. blocks-1
#             (no sigma after the last block)
#   T       = Q2 sigma( Q1 v_L )               project: width -> 64 -> 1
#
#   K_l v = IFFT( R_l . FFT(v) )   Fourier layer: keep only the lowest
#                                  modes_y x modes_x frequencies and multiply
#                                  each by a learned complex matrix R_l
#   W_l v                          pointwise 1x1 convolution (local mixing)
#
# The mask is NOT an input: it only enters the loss.
# =============================================================================

# =============================================================================
# VARIABLES (listed once)
#
#   width     : number of hidden channels (default 32)
#   modes_y   : retained Fourier modes along y (both + and - frequencies)
#   modes_x   : retained Fourier modes along x (rfft keeps only k_x >= 0)
#   blocks    : number of Fourier blocks (default 4)
#   padding   : zero points added on the top and right before the FFT, so
#               the non-periodic plate is not treated as periodic (default 8)
#   activation: "gelu" or "tanh"
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as functional


# =============================================================================
# FOURIER LAYER  K v = IFFT( R . FFT(v) )
#
# rfft2 of a real (Ny, Nx) field gives Ny x (Nx/2 + 1) complex coefficients:
#   rows 0 .. modes_y-1            : lowest positive k_y
#   rows Ny-modes_y .. Ny-1        : lowest negative k_y
#   columns 0 .. modes_x-1         : lowest k_x (only k_x >= 0 exist)
# Each of the two corner blocks gets its own weights R+ and R-.
# For every retained mode:  out_o = sum_i in_i R_io   (channel mixing)
# All other modes are set to zero (spectral truncation).
# =============================================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_y, modes_x):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_y = modes_y
        self.modes_x = modes_x

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_y, modes_x)

        self.weight_positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def multiply_modes(self, input_modes, weights):
        # (batch, in, ky, kx) x (in, out, ky, kx) -> (batch, out, ky, kx)
        return torch.einsum("bixy,ioxy->boxy", input_modes, weights)

    def forward(self, field):
        batch_size = field.shape[0]
        ny = field.shape[-2]
        nx = field.shape[-1]

        if self.modes_y * 2 > ny:
            raise ValueError("MODES_Y is too large for the grid height.")
        if self.modes_x > nx // 2 + 1:
            raise ValueError("MODES_X is too large for the grid width.")

        field_fourier = torch.fft.rfft2(field)

        output_fourier = torch.zeros(batch_size, self.out_channels, ny, nx // 2 + 1,
                                     dtype=torch.cfloat, device=field.device)

        my = self.modes_y
        mx = self.modes_x

        # POSITIVE k_y CORNER
        output_fourier[:, :, :my, :mx] = self.multiply_modes(
            field_fourier[:, :, :my, :mx], self.weight_positive)

        # NEGATIVE k_y CORNER
        output_fourier[:, :, -my:, :mx] = self.multiply_modes(
            field_fourier[:, :, -my:, :mx], self.weight_negative)

        return torch.fft.irfft2(output_fourier, s=(ny, nx))


class FNO2d(nn.Module):
    def __init__(self, input_channels, output_channels, modes_y, modes_x, width, padding,
                 blocks=4, activation="gelu"):
        super().__init__()
        self.padding = padding
        self.activation = activation

        # LIFT  P : input_channels -> width
        self.input_layer = nn.Conv2d(input_channels, width, kernel_size=1)

        # FOURIER BLOCKS  (K_l, W_l)
        self.spectral_layers = nn.ModuleList()
        self.local_layers = nn.ModuleList()
        for block in range(blocks):
            self.spectral_layers.append(SpectralConv2d(width, width, modes_y, modes_x))
            self.local_layers.append(nn.Conv2d(width, width, kernel_size=1))

        # PROJECT  Q1 : width -> 64,  Q2 : 64 -> output_channels
        self.output_layer_1 = nn.Conv2d(width, 64, kernel_size=1)
        self.output_layer_2 = nn.Conv2d(64, output_channels, kernel_size=1)

    def activate(self, field):
        if self.activation == "gelu":
            return functional.gelu(field)
        if self.activation == "tanh":
            return torch.tanh(field)
        raise ValueError("Activation must be tanh or gelu")

    def forward(self, field):
        field = self.input_layer(field)

        # PAD TOP AND RIGHT WITH ZEROS: (Ny, Nx) -> (Ny + padding, Nx + padding)
        if self.padding > 0:
            field = functional.pad(field, (0, self.padding, 0, self.padding))

        n_blocks = len(self.spectral_layers)
        for block in range(n_blocks):
            spectral_part = self.spectral_layers[block](field)
            local_part = self.local_layers[block](field)
            field = spectral_part + local_part

            # LAST FOURIER BLOCK STAYS LINEAR
            if block < n_blocks - 1:
                field = self.activate(field)

        # REMOVE THE PADDING
        if self.padding > 0:
            field = field[:, :, :-self.padding, :-self.padding]

        field = self.activate(self.output_layer_1(field))
        return self.output_layer_2(field)
