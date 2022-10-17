# MIT License

# Copyright (c) 2022 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Author: Hongrui Zheng
# Last Modified: 10/12/2022
# RBFLayer implemented in JAX/FLAX

from typing import Callable
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state

# RBF kernel functions
@jax.jit
def gaussian(alpha):
    phi = jnp.exp(-1 * alpha.pow(2))


@jax.jit
def inverse_quadratic(alpha):
    phi = jnp.ones_like(alpha) / (jnp.ones_like(alpha) + alpha.pow(2))


@jax.jit
def linear(alpha):
    phi = alpha
    return phi


@jax.jit
def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


@jax.jit
def multiquadric(alpha):
    phi = (jnp.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


@jax.jit
def inverse_multiquadric(alpha):
    phi = jnp.ones_like(alpha) / (jnp.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


@jax.jit
def spline(alpha):
    phi = alpha.pow(2) * jnp.log(alpha + jnp.ones_like(alpha))
    return phi


@jax.jit
def poisson_one(alpha):
    phi = (alpha - jnp.ones_like(alpha)) * jnp.exp(-alpha)
    return phi


@jax.jit
def poisson_two(alpha):
    phi = (
        ((alpha - 2 * jnp.ones_like(alpha)) / 2 * jnp.ones_like(alpha))
        * alpha
        * jnp.exp(-alpha)
    )
    return phi


@jax.jit
def matern32(alpha):
    phi = (jnp.ones_like(alpha) + 3**0.5 * alpha) * jnp.exp(-(3**0.5) * alpha)
    return phi


@jax.jit
def matern52(alpha):
    phi = (jnp.ones_like(alpha) + 5**0.5 * alpha + (5 / 3) * alpha.pow(2)) * jnp.exp(
        -(5**0.5) * alpha
    )
    return phi


class RBF(nn.Module):
    in_features: int
    num_kernels: int
    basis_func: Callable

    def setup(self):
        self.centers = self.param(
            "centers",
            nn.initializers.normal(1.0),
            (self.num_kernels, self.in_features),
        )
        self.log_sigs = self.param(
            "log_sigs",
            nn.initializers.constant(0.0),
            (self.num_kernels,),
        )

    def __call__(self, x):
        batch_size = x.shape[0]


class RBF(nn.Module)
    def __init__(self, in_features, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)