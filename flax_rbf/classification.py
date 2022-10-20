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
# Last Modified: 10/18/2022
# Classification example based on:
# https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/classification_demo.py

from flax_rbf import RBFNet
from flax_rbf import gaussian
# from flax.metrics import tensorboard
from flax.training import train_state
import optax
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# rng
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, num=3)

# GT decision boundary
x_gt = jnp.linspace(-1, 1, 101)
val_gt = 0.5*jnp.cos(jnp.pi*x_gt) + 0.5*jnp.cos(4*jnp.pi*(x_gt+1))

# random samples as training set
samples = 200
x1 = jax.random.uniform(k1, (samples, 1), minval=-1., maxval=1.)
x2_1 = jax.random.uniform(k2, (samples//2, 1), minval=-1., maxval=0.5*jnp.cos(jnp.pi*x1[:samples//2])+0.5*jnp.cos(4*jnp.pi*(x1[:samples//2]+1)))
x2_2 = jax.random.uniform(k3, (samples//2, 1), minval=0.5*jnp.cos(jnp.pi*x1[:samples//2])+0.5*jnp.cos(4*jnp.pi*(x1[:samples//2]+1)), maxval=1.)

# training set
tx = jnp.hstack((x1, jnp.vstack((x2_1, x2_2))))
ty = jnp.vstack((jnp.zeros((samples//2, 1)), jnp.ones((samples//2, 1))))

# gridding for plotting
steps = 100
x_span = jnp.linspace(-1, 1, steps)
y_span = jnp.linspace(-1, 1, steps)
xx, yy = jnp.meshgrid(x_span, y_span)
values = jnp.append(xx.ravel().reshape(xx.ravel().shape[0], 1),
                    yy.ravel().reshape(yy.ravel().shape[0], 1),
                    axis=1)

# create train state
rng = jax.random.PRNGKey(1)
rng, init_rng = jax.random.split(rng)

# Instanciating and training an RBF network with the Gaussian basis function
# This network receives a 2-dimensional input, transforms it into a 40-dimensional
# hidden representation with an RBF layer and then transforms that into a
# 1-dimensional output/prediction with a linear layer
rbf_net = RBFNet(in_features=2, out_features=1, num_kernels=40, basis_func=gaussian)
params = rbf_net.init(init_rng, jnp.ones((10, 2)))
optim = optax.adam(0.01)
tstate = train_state.TrainState.create(apply_fn=rbf_net.apply, params=params, tx=optim)


rbfnet = Network(layer_widths, layer_centres, basis_func)
rbfnet.fit(tx, ty, 5000, samples, 0.01, nn.BCEWithLogitsLoss())
rbfnet.eval()



# Plotting the ideal and learned decision boundaries

with torch.no_grad():
    preds = (torch.sigmoid(rbfnet(torch.from_numpy(values).float()))).data.numpy()
ideal_0 = values[np.where(values[:,1] <= 0.5*np.cos(np.pi*values[:,0]) + 0.5*np.cos(4*np.pi*(values[:,0]+1)))[0]]
ideal_1 = values[np.where(values[:,1] > 0.5*np.cos(np.pi*values[:,0]) + 0.5*np.cos(4*np.pi*(values[:,0]+1)))[0]]
area_0 = values[np.where(preds[:, 0] <= 0.5)[0]]
area_1 = values[np.where(preds[:, 0] > 0.5)[0]]

fig, ax = plt.subplots(figsize=(16,8), nrows=1, ncols=2)
ax[0].scatter(x[:samples//2,0], x[:samples//2,1], c='dodgerblue')
ax[0].scatter(x[samples//2:,0], x[samples//2:,1], c='orange', marker='x')
ax[0].scatter(ideal_0[:, 0], ideal_0[:, 1], alpha=0.1, c='dodgerblue')
ax[0].scatter(ideal_1[:, 0], ideal_1[:, 1], alpha=0.1, c='orange')
ax[0].set_xlim([-1,1])
ax[0].set_ylim([-1,1])
ax[0].set_title('Ideal Decision Boundary')
ax[1].scatter(x[:samples//2,0], x[:samples//2,1], c='dodgerblue')
ax[1].scatter(x[samples//2:,0], x[samples//2:,1], c='orange', marker='x')
ax[1].scatter(area_0[:, 0], area_0[:, 1], alpha=0.1, c='dodgerblue')
ax[1].scatter(area_1[:, 0], area_1[:, 1], alpha=0.1, c='orange')
ax[1].set_xlim([-1,1])
ax[1].set_ylim([-1,1])
ax[1].set_title('RBF Decision Boundary')
plt.show()