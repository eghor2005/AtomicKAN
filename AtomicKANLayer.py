import torch
import torch.nn as nn

from kernel import *


class AtomicKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, input_range=(-1, 1)):
        super(AtomicKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.input_range = input_range

        self.atomic_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.atomic_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

        self.register_buffer("centers", torch.linspace(input_range[0], input_range[1], degree + 1))

        center_spacing = (input_range[1] - input_range[0]) / degree
        compression_value = 1.0 / center_spacing  # Для 50% перекрытия

        self.compression = nn.Parameter(torch.ones(degree + 1) * compression_value)

        self.nsum = 100
        self.nprod = 10

    def forward(self, x):
        x = x.view(-1, self.inputdim, 1)
        centers = self.centers.view(1, 1, -1)
        compression = self.compression.view(1, 1, -1)
        scaled_x = (x - centers) * compression

        atomic_basis = fupn(scaled_x, n=1, nsum=self.nsum, nprod=self.nprod)
        y = torch.einsum("bid,iod->bo", atomic_basis, self.atomic_coeffs)
        return y.view(-1, self.outdim)

